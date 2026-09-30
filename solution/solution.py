"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    # Field order matters: tests build QAPair positionally as
    # (question, expected_answer, context, metadata).
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    # Keep the retriever's own ranking so the retrieval-side metrics can tell a
    # missed chunk (low recall) from a badly ordered one (low precision).
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    # Field order matters: tests build EvalResult positionally as
    # (qa_pair, actual_answer, faithfulness, relevance, completeness,
    #  passed, failure_type). The two retrieval fields come last so the
    # required positional construction keeps working, and both default to None
    # because they only exist when retrieved chunks were supplied.
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        The two retrieval metrics are deliberately excluded: they score the
        retriever on a different scale of meaning, and they are often None for
        some cases, so folding them in would produce a number that changes
        meaning depending on which subset of the run happened to supply chunks.
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _coverage(covered: set[str], total: set[str]) -> float:
    """Return |covered ∩ total| / |total|, clamped to [0, 1].

    Every answer-side and retrieval-side metric in this file is this one ratio
    with a different choice of numerator and denominator, so it lives in one
    place. Returns 1.0 for an empty denominator: there is nothing the score
    could fail to cover, and 0.0 would punish the caller for asking a vacuous
    question (e.g. an empty expected answer) rather than for a bad answer.
    """
    if not total:
        return 1.0
    return max(0.0, min(1.0, len(covered & total) / len(total)))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        return _coverage(_tokenize(context), _tokenize(answer))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(answer), _tokenize(question))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(answer), _tokenize(expected))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        union_tokens: set[str] = set()
        for chunk in contexts or []:
            union_tokens |= _tokenize(chunk)
        return _coverage(union_tokens, _tokenize(expected))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens or not contexts:
            return 1.0 if not expected_tokens else 0.0

        # A chunk counts as relevant once it carries enough of the expected
        # content. Chunks below the bar are noise for this question and act as
        # a precision penalty, not as a recall credit.
        relevant_flags = [
            len(_tokenize(chunk) & expected_tokens) / len(expected_tokens)
            >= relevance_threshold
            for chunk in contexts
        ]
        relevant_total = sum(relevant_flags)
        if relevant_total == 0:
            return 0.0

        # Running precision is only credited at the ranks that are relevant, so
        # a relevant chunk at rank 1 counts for far more than the same chunk at
        # rank 5.
        running_hits = 0
        weighted_sum = 0.0
        for rank, is_relevant in enumerate(relevant_flags, start=1):
            if not is_relevant:
                continue
            running_hits += 1
            weighted_sum += (running_hits / rank) * is_relevant

        return max(0.0, min(1.0, weighted_sum / relevant_total))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        # Ordered from most to least severe so one badly broken dimension
        # (e.g. an answer invented from nothing) is not masked by two fine ones.
        failure_type: str | None = None
        if faithfulness < 0.3:
            failure_type = "hallucination"
        elif relevance < 0.3:
            failure_type = "irrelevant"
        elif completeness < 0.3:
            failure_type = "incomplete"
        elif not passed:
            failure_type = "off_topic"
        if passed:
            failure_type = None

        # Retrieval metrics stay None when no chunks were supplied, so a caller
        # can always tell "not measured" apart from "measured and scored 0".
        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_recall=context_recall,
            context_precision=context_precision,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    query_tokens = _tokenize(query)
    # sorted() is stable, so chunks with equal overlap keep the retriever's own
    # ordering instead of being shuffled arbitrarily.
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.

    The judge is injected as a plain callable rather than an OpenAI client so
    that unit tests can drive it with a stub, and so the same class works
    against a local model, a different vendor, or a recorded transcript in
    production without touching the scoring code.
    """

    # Judges drift toward the middle, and the lecture rubric is written 1-5.
    # Anything above 1.0 is treated as a 1-5 answer and rescaled, which is
    # cheaper and more predictable than rejecting the response outright.
    SCALE_5_MAX = 5.0

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        criteria = "\n".join(f"- {name}: {description}" for name, description in rubric.items())
        prompt = (
            "You are grading one customer-support answer against a fixed rubric.\n"
            "Score every criterion on a 0.0-1.0 scale and reply with JSON only, in\n"
            "the form {\"scores\": {<criterion>: <float>}, \"reasoning\": \"<why>\"}.\n"
            "Judge only what the answer states. Ignore how long it is, and do not\n"
            "reward or punish it for matching a reference you were not given.\n\n"
            f"Criteria:\n{criteria or '- (no criteria supplied)'}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer to grade:\n{answer}\n"
        )

        raw = self.judge_llm_fn(prompt)
        parsed = self._parse_scores(raw)

        # Every requested criterion gets a number, even when the judge dropped
        # or renamed it. A missing criterion is a malformed judgement, not a
        # reason to silently shrink the rubric and change what is being measured.
        scores = {name: 0.5 for name in rubric}
        if parsed is not None:
            for name, value in parsed.items():
                if name in scores:
                    scores[name] = self._normalize(value)
        return {"scores": scores, "reasoning": str(raw).strip()}

    @staticmethod
    def _normalize(value: Any) -> float:
        """Coerce a judge score into [0, 1]."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        if 1.0 < number <= LLMJudge.SCALE_5_MAX:
            number = number / LLMJudge.SCALE_5_MAX
        return max(0.0, min(1.0, number))

    @staticmethod
    def _parse_scores(raw: Any) -> dict[str, float] | None:
        """Pull a criterion → number mapping out of a judge response.

        Accepts the documented {"scores": {...}} envelope, a bare {...} object,
        and the common case where the model wraps JSON in a ```json fence.
        Returns None when nothing usable is present, so the caller can fall back
        to neutral scores instead of crashing the benchmark.
        """
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        candidate = payload.get("scores", payload)
        if not isinstance(candidate, dict):
            return None
        numeric = {
            key: value
            for key, value in candidate.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return numeric or None

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        all_values: list[float] = []
        for item in scores_batch or []:
            if not isinstance(item, dict):
                continue
            criteria = item.get("scores")
            if not isinstance(criteria, dict):
                continue
            all_values.extend(
                self._normalize(value) for value in criteria.values()
            )

        mean_score = sum(all_values) / len(all_values) if all_values else 0.0
        leniency = bool(all_values) and mean_score > 0.8
        severity = bool(all_values) and mean_score < 0.3

        return {
            "positional_bias": self._detect_positional_bias(scores_batch or []),
            "leniency_bias": leniency,
            "severity_bias": severity,
            "mean_score": mean_score,
            "scored_items": len(all_values),
        }

    # A gap this small is inside the noise of a 20-sample batch, so calling it
    # bias would just add a warning nobody acts on.
    POSITIONAL_MARGIN = 0.05

    def _detect_positional_bias(self, scores_batch: list[dict[str, Any]]) -> bool:
        """Compare scores for the same case judged in slot 1 vs slot 2.

        Position bias is invisible in a single forward pass — the same answer
        scores differently depending on where it sits. So detection needs the
        paired design from Exercise 1.2: the same pair_id scored twice, once
        with this answer presented first and once second. Anything without that
        pairing is simply not evidence, and returns False rather than guessing.
        """
        by_pair: dict[Any, dict[str, float]] = {}
        for item in scores_batch:
            if not isinstance(item, dict):
                continue
            pair_id = item.get("pair_id", item.get("id"))
            position = item.get("position")
            if pair_id is None or position not in ("first", "second"):
                continue
            criteria = item.get("scores")
            if not isinstance(criteria, dict) or not criteria:
                continue
            by_pair.setdefault(pair_id, {})[str(position)] = sum(
                self._normalize(value) for value in criteria.values()
            ) / len(criteria)

        comparisons = [
            slots["first"] - slots["second"]
            for slots in by_pair.values()
            if "first" in slots and "second" in slots
        ]
        if not comparisons:
            return False
        # "Consistently" means it held on every paired case, not on average:
        # a bias that shows up on half the cases and reverses on the other half
        # is noise, not a systematic preference.
        return all(delta > self.POSITIONAL_MARGIN for delta in comparisons)


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            # run_full_eval only sees scalars, so it has to rebuild a QAPair.
            # Restoring the original keeps the id and difficulty that make a
            # failure traceable back to a specific case in the golden dataset.
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        passed = sum(1 for result in results if result.passed)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def mean_optional(attr: str) -> float | None:
            # Averaging over None values would silently treat "not measured" as
            # zero and drag the average down for the wrong reason.
            values = [
                value
                for value in (getattr(result, attr) for result in results)
                if value is not None
            ]
            return mean(values) if values else None

        failure_types: dict[str, int] = {}
        for result in results:
            label = result.failure_type or ("passed" if result.passed else "unspecified")
            failure_types[label] = failure_types.get(label, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
            "avg_faithfulness": mean([result.faithfulness for result in results]),
            "avg_relevance": mean([result.relevance for result in results]),
            "avg_completeness": mean([result.completeness for result in results]),
            "avg_overall": mean([result.overall_score() for result in results]),
            "avg_context_recall": mean_optional("context_recall"),
            "avg_context_precision": mean_optional("context_precision"),
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        TODO: Compute avg per metric, compare, list regressions, set passed flag
        """
        # The lecture's gate: any metric that drops by more than 0.05 against
        # the stored baseline is a regression, even if the absolute score still
        # looks acceptable. A pipeline that is only ever held to an absolute bar
        # ratchets downward one small loss at a time.
        drop_threshold = 0.05

        def average(results: list, attr: str) -> float:
            values = [float(getattr(result, attr)) for result in results]
            return sum(values) / len(values) if values else 0.0

        metrics = ("faithfulness", "relevance", "completeness")
        comparison: dict[str, Any] = {}
        regressions: list[str] = []
        for metric in metrics:
            new_avg = average(new_results, metric)
            baseline_avg = average(baseline_results, metric)
            delta = new_avg - baseline_avg
            comparison[f"new_avg_{metric}"] = new_avg
            comparison[f"baseline_avg_{metric}"] = baseline_avg
            comparison[f"delta_{metric}"] = delta
            if delta < -drop_threshold:
                regressions.append(metric)

        comparison["regressions"] = regressions
        comparison["drop_threshold"] = drop_threshold
        comparison["passed"] = not regressions
        return comparison

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            result
            for result in results
            if result.faithfulness < threshold
            or result.relevance < threshold
            or result.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.

    Both the suggestion list and the improvement log are organised by failure
    *cluster*, not by position in the input. Suggestions are produced once per
    cluster because one fix usually resolves many cases; pairing them with
    individual failures by index would attach an unrelated fix to most rows.
    """

    # The single fix that addresses each failure type, keyed by the same labels
    # run_full_eval() produces.
    FIX_BY_TYPE: dict[str, str] = {
        "hallucination": (
            "Gate generation on the retrieved evidence: reject any claim whose "
            "content words are absent from the retrieved chunks, and force an "
            "explicit 'insufficient evidence' answer instead of a guess"
        ),
        "irrelevant": (
            "Rewrite the system prompt to restate the question's intent and "
            "require the answer to address every part of it, then add the "
            "failing questions as few-shot examples"
        ),
        "incomplete": (
            "Check whether the expected answer is thinner than the gold "
            "evidence before touching the model; if the evidence is there, "
            "add an explicit per-condition checklist to the prompt"
        ),
        "off_topic": (
            "Add scope detection before generation so out-of-domain and "
            "false-premise requests are answered by the refusal path"
        ),
        "refusal": (
            "Loosen the guardrail on questions the corpus actually answers, "
            "and log each refusal for weekly review"
        ),
    }
    RETRIEVAL_FLOOR = 0.6

    def _fix_for(self, failure: EvalResult) -> str:
        """Return the fix that matches this specific failure.

        Retrieval is checked first: a case whose evidence was never retrieved
        cannot be fixed by changing the prompt, however low its answer-side
        scores look.
        """
        recall = failure.context_recall
        if recall is not None and recall < self.RETRIEVAL_FLOOR:
            return (
                "Retrieval missed the evidence (context recall "
                f"{recall:.2f}): widen top_k, add query expansion for policy "
                "terms, or switch to hybrid lexical + dense retrieval. Prompt "
                "changes cannot recover a chunk that was never retrieved"
            )
        return self.FIX_BY_TYPE.get(
            failure.failure_type or "unspecified",
            "Manual review required: this failure type has no mapped fix",
        )

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        categories: dict[str, int] = {}
        for failure in failures:
            label = failure.failure_type or "unspecified"
            categories[label] = categories.get(label, 0) + 1
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        # Which score is lowest is the useful signal: it names the stage that
        # broke. If the lowest score is not clearly worse than the others, no
        # single stage is to blame and the honest answer is the pipeline.
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        ranked = sorted(scores.items(), key=lambda item: item[1])
        lowest_value = ranked[0][1]
        tied = [name for name, value in ranked if value - lowest_value < 0.1]

        if len(tied) > 1:
            return "Multiple issues detected — review full pipeline"
        return {
            "faithfulness": "Context is missing or irrelevant — improve retrieval",
            "relevance": "Answer does not address the question — improve prompt clarity",
            "completeness": (
                "Answer is missing key information — increase context window "
                "or improve generation"
            ),
        }[ranked[0][0]]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".
        """
        header = (
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |\n"
            "|------------|------|------------|---------------|--------|"
        )
        rows: list[str] = []
        for index, failure in enumerate(failures, start=1):
            metadata = getattr(failure.qa_pair, "metadata", None) or {}
            failure_id = metadata.get("id") or f"F{index:03d}"
            # The cluster-level suggestion for THIS failure, not the suggestion
            # that happens to sit at the same index.
            suggestion = self._fix_for(failure)
            rows.append(
                "| {id} | {type} | {cause} | {fix} | Open |".format(
                    id=self._cell(failure_id),
                    type=self._cell(failure.failure_type or "unspecified"),
                    cause=self._cell(self.find_root_cause(failure)),
                    fix=self._cell(suggestion),
                )
            )
        return "\n".join([header, *rows])

    @staticmethod
    def _cell(text: Any) -> str:
        """Make a value safe to drop into a Markdown table cell."""
        # An unescaped pipe would silently split one row into two columns.
        return str(text).replace("|", "\\|").replace("\n", " ").strip()

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        # Ordered by how many cases each cluster explains, so the first
        # suggestion is the one that fixes the most failures per unit of work.
        ranked = sorted(categories.items(), key=lambda item: (-item[1], item[0]))

        suggestions: list[str] = []
        for label, count in ranked:
            template = self.FIX_BY_TYPE.get(label)
            if template:
                suggestions.append(f"{template} ({count} case(s) in this cluster)")

        # The two retrieval metrics are a separate diagnosis from the answer
        # metrics, so they get their own suggestions rather than being folded
        # into whichever generation cluster happened to be biggest.
        low_recall = [
            failure
            for failure in failures
            if failure.context_recall is not None
            and failure.context_recall < self.RETRIEVAL_FLOOR
        ]
        if low_recall:
            suggestions.append(
                f"Raise retrieval coverage for {len(low_recall)} low-recall "
                "case(s): widen top_k, add query expansion for policy terms, "
                "and verify no required paragraph is lost by chunking"
            )
        low_precision = [
            failure
            for failure in failures
            if failure.context_precision is not None
            and failure.context_precision < self.RETRIEVAL_FLOOR
        ]
        if low_precision:
            suggestions.append(
                f"Rerank before generation for {len(low_precision)} "
                "low-precision case(s): a cross-encoder reranker lifts rank-"
                "aware precision without changing the retrieved set"
            )

        suggestions.append(
            "Add every distinct failure case to the golden dataset so the next "
            "benchmark run can prove the fix worked instead of assuming it did"
        )
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
