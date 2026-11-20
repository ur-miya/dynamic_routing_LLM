"""Dependency-injected routing stages. No model loading or network setup."""
from __future__ import annotations

from src.generation.io_utils import messages
from src.generation.runtime_policy_v3 import route_one


SYSTEM_PROMPT = "You are a helpful, accurate, and concise assistant."
REASONS = {
    "finish_error": "student_error",
    "empty_answer": "empty_answer",
    "finish_length": "length_limit",
    "confidence_missing": "confidence_unavailable",
}


class StudentGenerationFailed(RuntimeError):
    """Explicit generation failure eligible for student_error fallback."""


class ConfidenceUnavailable(RuntimeError):
    """Explicit missing confidence eligible for confidence_unavailable fallback."""


class RuntimeStagesV3:
    def __init__(
        self, *, direct_score, generate, score_post, gate_score, teacher
    ):
        self.direct_score = direct_score
        self.generate = generate
        self.score_post = score_post
        self.gate_score = gate_score
        self.teacher = teacher

    def run(self, record, policy, *, prethreshold):
        method = policy["method"]
        direct = method.startswith("direct_")
        hybrid = method.startswith("hybrid_")
        if not direct and not method.startswith(("cascade_", "hybrid_")):
            raise ValueError("Unsupported routing method")

        allowed = set(policy.get("fallback_rules", []))
        if not direct and not allowed <= set(REASONS.values()):
            raise ValueError("Unknown fallback rule")

        calls = {
            "direct_score": 0,
            "student_generation": 0,
            "post_scoring": 0,
            "gate_score": 0,
            "teacher": 0,
        }
        generation = None
        post = None
        errors = []

        def finish(decision):
            teacher_result = None
            if decision["teacher"]:
                calls["teacher"] += 1
                teacher_result = self.teacher(
                    messages(record, include_history=True, system_prompt=SYSTEM_PROMPT)
                )
            return {
                **decision,
                "generation": generation,
                "post_features": post,
                "teacher_result": teacher_result,
                "calls": calls.copy(),
                "errors": errors.copy(),
            }

        if direct:
            score = None
            if not policy["threshold_policy"]["optional_disabled"]:
                calls["direct_score"] += 1
                score = self.direct_score(method, record)
            decision = route_one(policy, direct_score=score)
            if decision["student_needed"]:
                calls["student_generation"] += 1
                # Direct is not a cascade: do not silently add teacher fallback.
                generation = self.generate(
                    messages(record, include_history=True, system_prompt=SYSTEM_PROMPT)
                )
            return finish(decision)

        pre_score = None
        if hybrid:
            calls["direct_score"] += 1
            pre_score = self.direct_score("direct_LR_BE", record)
            threshold = policy.get("prethreshold", prethreshold)
            # route_one validates finiteness and applies the canonical comparison.
            pre_only = dict(policy)
            pre_only["method"] = "direct_LR_BE"
            pre_only["threshold_policy"] = {
                "optional_disabled": False, "threshold": threshold
            }
            if route_one(pre_only, direct_score=pre_score)["teacher"]:
                return finish(route_one(
                    policy, pre_score=pre_score, prethreshold=prethreshold
                ))

        flags = {
            "finish_error": 0,
            "empty_answer": 0,
            "finish_length": 0,
            "confidence_missing": 0,
        }
        calls["student_generation"] += 1
        try:
            generation = self.generate(
                messages(record, include_history=True, system_prompt=SYSTEM_PROMPT)
            )
        except StudentGenerationFailed as exc:
            flags.update(finish_error=1, empty_answer=1, confidence_missing=1)
            errors.append({"stage": "generation", "error": str(exc)})
        else:
            text = generation["text"]
            if not isinstance(text, str):
                raise TypeError("Generation text must be str")
            finish_reason = str(generation["finish_reason"]).lower()
            status = str(generation["status"]).lower()
            flags["empty_answer"] = int(not text.strip())
            flags["finish_length"] = int(finish_reason == "length")
            flags["finish_error"] = int(
                status != "ok" or finish_reason not in {"stop", "eos", "length"}
            )

        def mandatory():
            return any(
                bool(flags[field]) and reason in allowed
                for field, reason in REASONS.items()
            )

        def route(gate=None):
            return route_one(
                policy, pre_score=pre_score, gate_score=gate,
                flags=flags, prethreshold=prethreshold,
            )

        if mandatory():
            return finish(route())

        if generation is None:
            raise RuntimeError("Generation failed without an allowed fallback")

        calls["post_scoring"] += 1
        try:
            post = self.score_post(generation)
        except ConfidenceUnavailable as exc:
            flags["confidence_missing"] = 1
            errors.append({"stage": "scoring", "error": str(exc)})
            if mandatory():
                return finish(route())
            raise

        for field in REASONS:
            value = post[field]
            if value not in (0, 1, False, True):
                raise ValueError(f"Invalid post flag: {field}")
            if field != "confidence_missing" and int(value) != flags[field]:
                raise ValueError(f"Generation/post flag disagreement: {field}")
            flags[field] = int(value)

        if mandatory():
            return finish(route())

        if policy["threshold_policy"]["optional_disabled"]:
            return finish(route())

        calls["gate_score"] += 1
        score = self.gate_score(record, post)
        return finish(route(score))
