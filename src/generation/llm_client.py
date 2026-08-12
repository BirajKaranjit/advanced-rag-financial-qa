"""Swappable LLM client: Groq or Gemini as the primary provider (selected
via GENERATION_PROVIDER in .env), Hugging Face Inference as a documented
last-resort fallback for plain-text completions.

Groq is the documented default for its free tier, low latency, and
mature tool-calling support. Gemini (Google AI Studio's free tier) is a
documented alternative for users who'd rather use a Gemini key -- both
`complete` and `complete_with_tools` work against either provider; only
the wire format differs internally.
"""

from __future__ import annotations

import json
import logging
import httpx
from typing import Any

from groq import Groq
from huggingface_hub import InferenceClient

from config import settings
from src.exceptions import GenerationError

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]

_orig_httpx_init = httpx.Client.__init__

def _patched_httpx_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    _orig_httpx_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_init

def _model_access_guidance(provider: str, model: str) -> str:
    if provider == "gemini":
        return (
            f"Configured Gemini model '{model}' may be unavailable for this API key. "
            "Set GEMINI_MODEL in .env to a model listed in your Google AI Studio account "
            "for that key."
        )
    if provider == "groq":
        return (
            f"Configured Groq model '{model}' may be unavailable for this API key. "
            "Set GROQ_MODEL in .env to a model listed in your Groq account for that key."
        )
    if provider == "hf":
        return (
            f"Configured HF fallback model '{model}' may be unavailable or gated. "
            "Set HF_FALLBACK_MODEL in .env to a model your token can access."
        )
    return ""


def _gemini_schema_type(type_name: str) -> Any:
    """Map OpenAI JSON schema types to Gemini proto enum values."""
    if genai is None:
        return None
    type_map = {
        "string": genai.protos.Type.STRING,
        "number": genai.protos.Type.NUMBER,
        "integer": genai.protos.Type.INTEGER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
        "object": genai.protos.Type.OBJECT,
    }
    return type_map.get(type_name, genai.protos.Type.TYPE_UNSPECIFIED)


def _json_schema_to_gemini_schema(schema: Any) -> Any:
    """Recursively convert OpenAI JSON schema into a Gemini Schema proto."""
    if genai is None:
        return schema
    if isinstance(schema, genai.protos.Schema):
        return schema
    if not isinstance(schema, dict):
        schema = {"type": "object"}

    schema_type = _gemini_schema_type(str(schema.get("type", "object")).lower())
    converted: dict[str, Any] = {"type": schema_type}
    if "description" in schema:
        converted["description"] = schema["description"]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        converted["properties"] = {
            key: _json_schema_to_gemini_schema(value) for key, value in properties.items()
        }
    items = schema.get("items")
    if isinstance(items, dict):
        converted["items"] = _json_schema_to_gemini_schema(items)
    if "enum" in schema:
        converted["enum"] = list(schema["enum"])
    if "required" in schema:
        converted["required"] = list(schema["required"])
    return genai.protos.Schema(**converted)


def _openai_tool_to_gemini_declaration(tool_schema: dict) -> Any:
    """Convert an OpenAI-style tool schema into Gemini FunctionDeclaration."""
    fn = tool_schema["function"]
    parameters = fn.get("parameters") or {"type": "object", "properties": {}}
    return genai.protos.FunctionDeclaration(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=_json_schema_to_gemini_schema(parameters),
    )


def _normalize_gemini_function_args(args: Any) -> dict[str, Any]:
    """Coerce Gemini MapComposite inputs to a plain dict for tool execution."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if hasattr(args, "items"):
        try:
            return dict(args.items())
        except Exception:  # pragma: no cover - defensive fallback
            pass
    return {}


def _safe_get_gemini_text(response: Any) -> str:
    """Safely extract text from a Gemini response without throwing when function calls or non-text parts are present."""
    try:
        return response.text or ""
    except (ValueError, AttributeError):
        pass

    text_parts = []
    candidates = getattr(response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
    return "".join(text_parts)


class LlmClient:
    """Thin provider-agnostic wrapper. `complete` is used for plain text
    generation (rewrites, HyDE, final answers without tools);
    `complete_with_tools` is used for the generation step that may call
    the numeric-store tool. Both dispatch to the configured
    `generation_provider`, falling back to Hugging Face Inference (plain
    text only) if the primary provider call fails.
    """

    def __init__(self) -> None:
        self.provider = settings.generation_provider
        self._groq = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None
        self._hf = (
            InferenceClient(token=settings.hf_inference_token)
            if settings.hf_inference_token
            else None
        )
        self._gemini_configured = False
        if settings.gemini_api_key and genai is not None:
            genai.configure(api_key=settings.gemini_api_key)
            self._gemini_configured = True
        elif settings.gemini_api_key and genai is None:
            logger.warning(
                "GEMINI_API_KEY is set but google-generativeai is not installed; "
                "run `pip install google-generativeai`."
            )

    # -- plain completion -----------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        primary = self._complete_gemini if self.provider == "gemini" else self._complete_groq
        primary_name = "Gemini" if self.provider == "gemini" else "Groq"
        try:
            return primary(system_prompt, user_prompt, max_tokens)
        except Exception as primary_exc:  # noqa: BLE001
            logger.warning("%s completion failed, trying HF fallback: %s", primary_name, primary_exc)
            try:
                return self._complete_hf(system_prompt, user_prompt, max_tokens)
            except Exception as fallback_exc:  # noqa: BLE001
                raise GenerationError(
                    f"Both {primary_name} and HF fallback failed: {primary_exc} / {fallback_exc}"
                ) from fallback_exc

    def _complete_groq(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        if self._groq is None:
            raise GenerationError("Groq client not configured; set GROQ_API_KEY")
        try:
            response = self._groq.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens or settings.generation_max_tokens,
                temperature=settings.generation_temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            guidance = _model_access_guidance("groq", settings.groq_model)
            raise GenerationError(f"Groq generation failed: {exc}. {guidance}") from exc

    def _complete_gemini(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        if not self._gemini_configured:
            raise GenerationError("Gemini client not configured; set GEMINI_API_KEY")
        try:
            model = genai.GenerativeModel(settings.gemini_model, system_instruction=system_prompt)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens or settings.generation_max_tokens,
                    temperature=settings.generation_temperature,
                ),
            )
            return _safe_get_gemini_text(response)
        except Exception as exc:  # noqa: BLE001
            guidance = _model_access_guidance("gemini", settings.gemini_model)
            raise GenerationError(f"Gemini generation failed: {exc}. {guidance}") from exc

    def _complete_hf(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        if self._hf is None:
            raise GenerationError("HF Inference client not configured; set HF_INFERENCE_TOKEN")
        try:
            prompt = f"<system>{system_prompt}</system>\n<user>{user_prompt}</user>"
            return self._hf.text_generation(
                prompt, model=settings.hf_fallback_model, max_new_tokens=max_tokens or 512
            )
        except Exception as exc:  # noqa: BLE001
            guidance = _model_access_guidance("hf", settings.hf_fallback_model)
            raise GenerationError(f"HF generation failed: {exc}. {guidance}") from exc

    # -- tool-calling completion --------------------------------------------------

    def complete_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_executor,
        max_tokens: int | None = None,
    ) -> tuple[str, bool, str | None]:
        """Runs a single-round tool-calling loop against the configured provider."""
        if self.provider == "gemini":
            return self._complete_with_tools_gemini(
                system_prompt, user_prompt, tools, tool_executor, max_tokens
            )
        if self.provider == "groq":
            return self._complete_with_tools_groq(
                system_prompt, user_prompt, tools, tool_executor, max_tokens
            )
        logger.warning(
            "HF provider selected: tool calling is disabled, falling back to plain completion."
        )
        return self.complete(system_prompt, user_prompt, max_tokens=max_tokens), False, None

    def _complete_with_tools_groq(
        self, system_prompt, user_prompt, tools, tool_executor, max_tokens
    ) -> tuple[str, bool, str | None]:
        if self._groq is None:
            raise GenerationError("Groq client not configured; set GROQ_API_KEY")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._groq.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens or settings.generation_max_tokens,
                temperature=settings.generation_temperature,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or "", False, None

            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            tool_result = tool_executor(tool_call.function.name, args)

            messages.append(message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )
            follow_up = self._groq.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=max_tokens or settings.generation_max_tokens,
                temperature=settings.generation_temperature,
            )
            final_text = follow_up.choices[0].message.content or ""
            return final_text, True, str(tool_result)
        except Exception as exc:  # noqa: BLE001
            guidance = _model_access_guidance("groq", settings.groq_model)
            raise GenerationError(
                f"Groq tool-calling generation failed: {exc}. {guidance}"
            ) from exc

    def _complete_with_tools_gemini(
        self, system_prompt, user_prompt, tools, tool_executor, max_tokens
    ) -> tuple[str, bool, str | None]:
        if not self._gemini_configured:
            raise GenerationError("Gemini client not configured; set GEMINI_API_KEY")
        try:
            gemini_declarations = [_openai_tool_to_gemini_declaration(t) for t in tools]
            gemini_tools = [genai.protos.Tool(function_declarations=gemini_declarations)]
            model = genai.GenerativeModel(
                settings.gemini_model,
                system_instruction=system_prompt,
                tools=gemini_tools,
            )
            chat = model.start_chat()
            response = chat.send_message(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens or settings.generation_max_tokens,
                    temperature=settings.generation_temperature,
                ),
            )

            function_call = None
            if response.candidates and response.candidates[0].content:
                for part in getattr(response.candidates[0].content, "parts", []):
                    candidate_call = getattr(part, "function_call", None)
                    if candidate_call is not None and getattr(candidate_call, "name", ""):
                        function_call = candidate_call
                        break

            if function_call is None:
                return _safe_get_gemini_text(response), False, None

            tool_name = getattr(function_call, "name", "")
            tool_args = _normalize_gemini_function_args(getattr(function_call, "args", None))
            tool_result = tool_executor(tool_name, tool_args)

            follow_up = chat.send_message(
                genai.protos.Content(
                    parts=[
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name, response={"result": str(tool_result)}
                            )
                        )
                    ]
                )
            )
            return _safe_get_gemini_text(follow_up), True, str(tool_result)
        except Exception as exc:  # noqa: BLE001
            guidance = _model_access_guidance("gemini", settings.gemini_model)
            raise GenerationError(
                f"Gemini tool-calling generation failed: {exc}. {guidance}"
            ) from exc