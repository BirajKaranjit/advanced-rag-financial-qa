"""CLI entry point: run the evaluation harness against an ingested
document and print a per-question-type metrics summary.

Assumes `scripts/ingest.py` has already been run against the target PDF.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.evaluation.eval_dataset import get_eval_examples
from src.evaluation.metrics import aggregate_by_question_type, score_example
from src.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument(
        "--mode", choices=["basic", "advanced"], default="advanced",
        help="Which pipeline mode to evaluate.",
    )
    parser.add_argument(
        "--output", default="data/processed/eval_results.json",
        help="Where to write per-example results as JSON.",
    )
    args = parser.parse_args()

    pipeline = RagPipeline(mode=args.mode)
    examples = get_eval_examples()

    results = []
    for example in examples:
        logger.info("Evaluating %s: %s", example.query_id, example.question)
        outcome = pipeline.ask(example.question)
        retrieved_ids = [h.chunk_id for h in outcome.trace.reranked_hits] or [
            h.chunk_id for h in outcome.trace.dense_hits
        ]
        # Answer correctness requires either human judgment or an LLM
        # judge; left as None here and filled in by a separate manual
        # or judged pass -- see ARCHITECTURE.md, Evaluation methodology.
        results.append(score_example(example, retrieved_ids, outcome.answer, answer_correct=None))

    summary = aggregate_by_question_type(results)
    print(json.dumps(summary, indent=2))

    with open(args.output, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)
    logger.info("Wrote per-example results to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
