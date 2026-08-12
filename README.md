# Advanced Hybrid RAG — Financial Filing Q&A

A hybrid retrieval-augmented generation system for answering natural-language
questions against dense, table-heavy business PDFs (SEC filings, invoices,
financial reports). Combines structure-aware parent/child chunking, a
normalized numeric store for exact computation, dense + sparse hybrid
retrieval with reranking, and a production-hardening layer (observability,
table-checksum validation, bounding-box lineage, prompt-injection defenses,
and a numeric groundedness check).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design writeup,
diagrams, threat model, and evaluation results.

## Setup — you need exactly two keys

1. **`HUGGINGFACE_HUB_TOKEN`** — authenticates downloads of the local
   embedding/reranker models (`bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2`).
   Both are public models, so this is technically optional, but anonymous
   Hugging Face Hub downloads are rate-limited and this avoids first-run
   failures. Get one free at <https://huggingface.co/settings/tokens>.
2. **One LLM key** — either:
   - `GROQ_API_KEY` (default provider, free tier: <https://console.groq.com>), or
   - `GEMINI_API_KEY` (free tier: <https://aistudio.google.com/apikey>) —
     set `GENERATION_PROVIDER=gemini` in `.env` if you use this instead.

```bash
git clone <repo-url>
cd advanced-rag-financial-qa
cp .env.example .env
# edit .env: set HUGGINGFACE_HUB_TOKEN and either GROQ_API_KEY or
# GEMINI_API_KEY (+ GENERATION_PROVIDER=gemini if using Gemini)
```

## Run with Docker (recommended)

One command, no local Python/system-dependency setup required. The image
bundles `poppler-utils` (pdfplumber's PDF backend) and `tesseract-ocr` (the
scanned-page OCR fallback) so it runs identically on any machine with Docker
installed. If using Windows, make sure Docker Desktop Engine is running.
```bash
docker compose up --build
```

Then open <http://localhost:8501>. Ingested indexes persist in `./data` on
the host across container restarts (mounted as a volume in
`docker-compose.yml`), so you don't need to re-ingest after a rebuild.

## Run locally with pip (alternative)

```bash
pip install -r requirements.txt

# System dependencies (skip this if you're using Docker):
#   Debian/Ubuntu: sudo apt-get install poppler-utils tesseract-ocr
#   macOS:         brew install poppler tesseract

streamlit run app.py
```

## Using the UI

Upload a PDF in the sidebar, click **Ingest document**, then ask questions
in the chat panel. Toggle between "Basic RAG" and "Advanced Hybrid RAG" to
compare retrieval quality side by side. Expand the **Retrieval trace** panel
under any answer to see dense hits, BM25 hits, the RRF-fused ranking,
reranked results, which parent chunks were expanded, per-stage timings, any
chunks excluded for prompt-injection risk, and the numeric groundedness
result.

## Run from the command line

```bash
# Ingest a document
python scripts/ingest.py data/raw/2022_Q3_AAPL.pdf

# Run the evaluation harness
python scripts/evaluate.py --mode advanced
```

## Run tests / lint (also run automatically in CI, see .github/workflows/ci.yml)

```bash
pytest
ruff check .
black --check .
mypy src/ --ignore-missing-imports
```

## Repository structure

```
advanced-rag-financial-qa/
├── Dockerfile                  # multi-stage build (poppler + tesseract runtime)
├── docker-compose.yml          # one-command startup, persists ./data
├── .github/workflows/ci.yml    # lint, type-check, test, docker build
├── config.py                   # pydantic-settings configuration
├── app.py                      # Streamlit UI
├── src/
│   ├── schemas.py               # shared pydantic models
│   ├── exceptions.py            # custom exception hierarchy
│   ├── ingestion/                 # PDF parsing, metadata, chunking, checksum validation
│   ├── indexing/                   # embeddings, ChromaDB, BM25, SQLite store
│   ├── retrieval/                   # router, query transform, hybrid search,
│   │                                 # RRF fusion, reranker, compressor, tracer
│   ├── generation/                   # LLM client (Groq/Gemini), numeric-store
│   │                                 # tool, prompts, groundedness check
│   ├── security/                      # prompt-injection scanning + sanitization
│   ├── evaluation/                    # labeled eval set + metrics
│   └── pipeline.py                    # end-to-end orchestration
├── scripts/                     # ingest.py, evaluate.py CLIs
└── tests/                       # chunker, RRF fusion, numeric-store, checksum,
                                  # prompt-injection, groundedness tests
```

## Notes

- No document-specific logic is hardcoded; the pipeline is designed to work
  against other financial/business PDFs of similar density without code
  changes.
- The generation provider is swappable via `GENERATION_PROVIDER` in `.env`
  (Groq or Gemini), with Hugging Face Inference as a last-resort plain-text
  fallback if the primary provider call fails.
- Ingested content is treated as untrusted input throughout generation. See
  ARCHITECTURE.md, "Prompt injection and RAG document-injection defenses."
