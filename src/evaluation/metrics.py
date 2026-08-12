"""Retrieval and answer evaluation metrics."""

from __future__ import annotations

from collections import defaultdict

from src.schemas import EvalExample, EvalQuestionType, EvalResult


def hit_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> bool:
    if not ground_truth_ids:
        return False
    return any(cid in ground_truth_ids for cid in retrieved_ids[:k])


def mean_reciprocal_rank(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    if not ground_truth_ids:
        return 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def score_example(
    example: EvalExample, retrieved_ids: list[str], predicted_answer: str, answer_correct: bool | None
) -> EvalResult:
    return EvalResult(
        query_id=example.query_id,
        question_type=example.question_type,
        predicted_answer=predicted_answer,
        retrieved_chunk_ids=retrieved_ids,
        hit_at_5=hit_at_k(retrieved_ids, example.ground_truth_chunk_ids, k=5),
        mrr=mean_reciprocal_rank(retrieved_ids, example.ground_truth_chunk_ids),
        answer_correct=answer_correct,
    )


def aggregate_by_question_type(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """Groups results by question_type and computes mean hit@5, mean MRR,
    and answer accuracy (when judged) per group.
    """
    grouped: dict[EvalQuestionType, list[EvalResult]] = defaultdict(list)
    for result in results:
        grouped[result.question_type].append(result)

    summary: dict[str, dict[str, float]] = {}
    for question_type, group in grouped.items():
        n = len(group)
        judged = [r for r in group if r.answer_correct is not None]
        summary[question_type.value] = {
            "n": n,
            "hit_at_5": sum(r.hit_at_5 for r in group) / n if n else 0.0,
            "mrr": sum(r.mrr for r in group) / n if n else 0.0,
            "answer_accuracy": (
                sum(bool(r.answer_correct) for r in judged) / len(judged) if judged else float("nan")
            ),
        }
    return summary
