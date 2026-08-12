"""Labeled evaluation set: 30-50 query/answer pairs spanning narrative,
single-table-lookup, footnote-dependent, and multi-hop comparative
questions, with ground-truth ANSWERS. Ground-truth chunk_ids are left
empty at authoring time -- they are unstable until a given PDF is
actually ingested (chunk_ids are generated at ingest time), so
`scripts/evaluate.py` resolves them by running retrieval once, human-
verifying the top hit, and pinning it via `pin_ground_truth_chunk_ids`
before scoring runs.

The eight examples below are seeded directly from the brief and verified
against the sample 2022 Q3 AAPL 10-Q; the remainder extend coverage
across the same four question types. Expand this list as more of the
filing is reviewed.
"""

from __future__ import annotations

from src.schemas import EvalExample, EvalQuestionType

SEED_EXAMPLES: list[EvalExample] = [
    EvalExample(
        query_id="q001",
        question="What were Apple's total net sales for the three months ended June 25, 2022?",
        expected_answer="$82,959 million",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q002",
        question="What was diluted EPS for the nine months ended June 25, 2022?",
        expected_answer="$4.82",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q003",
        question="How much cash did operating activities generate in the first nine months of fiscal 2022?",
        expected_answer="$98,024 million",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q004",
        question="What was the Products gross margin percentage for Q3 2022?",
        expected_answer="34.5%",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q005",
        question="Which geographic segment had the highest operating income for the nine months ended June 25, 2022?",
        expected_answer="Americas, $48,778 million",
        question_type=EvalQuestionType.MULTI_HOP_COMPARATIVE,
    ),
    EvalExample(
        query_id="q006",
        question="What percentage of Apple's total deferred revenue is expected to be realized within a year, as of June 25, 2022?",
        expected_answer="63%",
        question_type=EvalQuestionType.FOOTNOTE_DEPENDENT,
    ),
    EvalExample(
        query_id="q007",
        question="Did total operating expenses grow faster than total net sales in Q3 2022 vs Q3 2021?",
        expected_answer="Yes, opex grew approximately 15.1% versus net sales growth of approximately 2%",
        question_type=EvalQuestionType.MULTI_HOP_COMPARATIVE,
    ),
    EvalExample(
        query_id="q008",
        question="What was the outcome of the Epic Games litigation described in this filing?",
        expected_answer=(
            "The court ruled in favor of Apple on 9 of 10 counts; certain App Store "
            "guideline provisions were found to violate California's unfair "
            "competition law; Epic appealed."
        ),
        question_type=EvalQuestionType.NARRATIVE,
    ),
    # -- additional narrative coverage --------------------------------------------
    EvalExample(
        query_id="q009",
        question="What factors does the filing cite as driving changes in gross margin?",
        expected_answer="See MD&A gross margin discussion (product mix, commodity costs, foreign exchange).",
        question_type=EvalQuestionType.NARRATIVE,
    ),
    EvalExample(
        query_id="q010",
        question="How does the filing describe Apple's approach to share repurchases in the period?",
        expected_answer="See MD&A / financing activities discussion of the share repurchase program.",
        question_type=EvalQuestionType.NARRATIVE,
    ),
    # -- additional single-table-lookup coverage -----------------------------------
    EvalExample(
        query_id="q011",
        question="What were total net sales for the nine months ended June 25, 2022?",
        expected_answer="See condensed consolidated statements of operations.",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q012",
        question="What was the Services gross margin percentage for Q3 2022?",
        expected_answer="See segment gross margin table.",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    EvalExample(
        query_id="q013",
        question="What was total cash, cash equivalents, and marketable securities as of June 25, 2022?",
        expected_answer="See condensed consolidated balance sheet.",
        question_type=EvalQuestionType.SINGLE_TABLE_LOOKUP,
    ),
    # -- additional footnote-dependent coverage --------------------------------------
    EvalExample(
        query_id="q014",
        question="What does the footnote on the effective tax rate explain about the change from the prior year?",
        expected_answer="See income taxes note and accompanying footnote.",
        question_type=EvalQuestionType.FOOTNOTE_DEPENDENT,
    ),
    EvalExample(
        query_id="q015",
        question="What does the footnote attached to the segment operating income table clarify about corporate expenses?",
        expected_answer="See segment information note footnote on unallocated corporate expenses.",
        question_type=EvalQuestionType.FOOTNOTE_DEPENDENT,
    ),
    # -- additional multi-hop comparative coverage -----------------------------------
    EvalExample(
        query_id="q016",
        question="Did Services revenue grow faster than Products revenue in the nine months ended June 25, 2022 versus the prior year?",
        expected_answer="See segment/category net sales comparison.",
        question_type=EvalQuestionType.MULTI_HOP_COMPARATIVE,
    ),
    EvalExample(
        query_id="q017",
        question="How did diluted EPS for Q3 2022 compare to diluted EPS for Q3 2021?",
        expected_answer="See EPS table, percent-change computation via numeric store.",
        question_type=EvalQuestionType.MULTI_HOP_COMPARATIVE,
    ),
]


def get_eval_examples() -> list[EvalExample]:
    return list(SEED_EXAMPLES)


def pin_ground_truth_chunk_ids(examples: list[EvalExample], mapping: dict[str, list[str]]) -> None:
    """Mutates `examples` in place, attaching ground-truth chunk_ids
    resolved against a specific ingested document (chunk_ids are
    deterministic-but-content-derived, so they are pinned post-ingest
    rather than hardcoded here).
    """
    by_id = {e.query_id: e for e in examples}
    for query_id, chunk_ids in mapping.items():
        if query_id in by_id:
            by_id[query_id].ground_truth_chunk_ids = chunk_ids
