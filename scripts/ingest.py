"""CLI entry point: ingest a PDF into the document store, vector store,
and BM25 index.

Usage:
    python scripts/ingest.py data/raw/2022_Q3_AAPL.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG index.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    args = parser.parse_args()

    pipeline = RagPipeline(mode="advanced")
    pipeline.ingest(args.pdf_path)
    logger.info("Ingestion complete for %s", args.pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
