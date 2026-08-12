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


def _openai_tool_to_gemini_declaration(tool_schema: dict) -> dict:
    """Converts one OpenAI-style tool schema (as used for Groq) into the
    function-declaration shape google-generativeai expects.

    Note: the google-generativeai SDK's accepted `tools=` shape has moved
    across releases (raw dict vs. genai.types.Tool vs. protos.Tool). This
    passes a plain dict, the form documented as accepted directly by
    `GenerativeModel(tools=...)` at the time of writing; if a future SDK
    version rejects it, wrap the return value in
    `genai.types.Tool(function_declarations=[...])` instead.
    """
    fn = tool_schema["function"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
    }


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

    def _complete_gemini(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        if not self._gemini_configured:
            raise GenerationError("Gemini client not configured; set GEMINI_API_KEY")
        model = genai.GenerativeModel(settings.gemini_model, system_instruction=system_prompt)
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens or settings.generation_max_tokens,
                temperature=settings.generation_temperature,
            ),
        )
        return response.text or ""

    def _complete_hf(self, system_prompt: str, user_prompt: str, max_tokens: int | None) -> str:
        if self._hf is None:
            raise GenerationError("HF Inference client not configured; set HF_INFERENCE_TOKEN")
        prompt = f"<system>{system_prompt}</system>\n<user>{user_prompt}</user>"
        return self._hf.text_generation(
            prompt, model=settings.hf_fallback_model, max_new_tokens=max_tokens or 512
        )

    # -- tool-calling completion --------------------------------------------------

    def complete_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_executor,
        max_tokens: int | None = None,
    ) -> tuple[str, bool, str | None]:
        """Runs a single-round tool-calling loop against the configured
        provider (Groq or Gemini).

        Returns:
            (final_answer_text, tool_was_called, tool_result_text)
        """
        if self.provider == "gemini":
            return self._complete_with_tools_gemini(
                system_prompt, user_prompt, tools, tool_executor, max_tokens
            )
        return self._complete_with_tools_groq(
            system_prompt, user_prompt, tools, tool_executor, max_tokens
        )

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
            raise GenerationError(f"Groq tool-calling generation failed: {exc}") from exc

    def _complete_with_tools_gemini(
        self, system_prompt, user_prompt, tools, tool_executor, max_tokens
    ) -> tuple[str, bool, str | None]:
        if not self._gemini_configured:
            raise GenerationError("Gemini client not configured; set GEMINI_API_KEY")
        try:
            gemini_tools = [_openai_tool_to_gemini_declaration(t) for t in tools]
            model = genai.GenerativeModel(
                settings.gemini_model, system_instruction=system_prompt, tools=gemini_tools
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
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    function_call = part.function_call
                    break

            if function_call is None:
                return response.text or "", False, None

            args = dict(function_call.args)
            tool_result = tool_executor(function_call.name, args)

            follow_up = chat.send_message(
                genai.protos.Content(
                    parts=[
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_call.name, response={"result": str(tool_result)}
                            )
                        )
                    ]
                )
            )
            return follow_up.text or "", True, str(tool_result)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Gemini tool-calling generation failed: {exc}") from exc
