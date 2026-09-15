"""Turn orchestration.

One user reply flows through here: extract -> merge -> score -> respond. The
ordering matters and is the reason the score can never be a model opinion — by
the time the conversational call happens, the number already exists and is
handed to the model as a fact it must narrate rather than produce.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .confidence import compute_confidence, confidence_label
from .llm import LLMClient, LLMUnavailable
from .schemas import SLOT_NAMES, ProfileExtraction, Recommendation
from .state import SessionState


@dataclass
class TurnResult:
    text: str
    score: float
    delta: float
    label: str
    recommendation: Recommendation | None = None
    is_final: bool = False


class Conversation:
    def __init__(self, client: LLMClient, state: SessionState | None = None) -> None:
        self.client = client
        self.state = state or SessionState()

    # ------------------------------------------------------------------ utils

    def _slot_for_current_question(self) -> str:
        index = max(0, min(self.state.current_question_index - 1, len(SLOT_NAMES) - 1))
        return SLOT_NAMES[index]

    def _api_messages(self) -> list[dict[str, str]]:
        """Transcript rendered for the API, prefixed with the control token."""
        messages = [{"role": "user", "content": prompts.KICKOFF_MESSAGE}]
        messages.extend({"role": t.role, "content": t.content} for t in self.state.transcript)
        return messages

    def _transcript_text(self) -> str:
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in self.state.transcript)

    def _rescore(self) -> tuple[float, float]:
        score = compute_confidence(
            self.state.profile,
            self.state.contradictions,
            self.state.declined_slots,
            measurements_generic_only=self.state.measurements_generic_only,
        )
        delta = self.state.record_confidence(score)
        return score, delta

    # ------------------------------------------------------------------ turns

    def open(self) -> TurnResult:
        """The first assistant turn. Model-authored, like every other one."""
        system = prompts.build_system_prompt(
            self.state.summary_for_prompt(), prompts.TASK_ASK_NEXT_V1
        )
        text = self.client.call(
            system,
            [{"role": "user", "content": prompts.KICKOFF_MESSAGE}],
            call_type="open",
        )
        self.state.add_turn("assistant", str(text))
        self.state.current_question_index = 1
        score, delta = self._rescore()
        return TurnResult(text=str(text), score=score, delta=delta, label=confidence_label(score))

    def handle(self, user_text: str) -> TurnResult:
        """Process one user reply end to end."""
        slot = self._slot_for_current_question()
        user_turn = self.state.add_turn("user", user_text)

        try:
            extraction = self._extract(user_text, slot)
        except LLMUnavailable:
            # Roll the turn back so a retry does not duplicate the user's words.
            self.state.transcript.remove(user_turn)
            raise

        self.state.merge_extraction(extraction)
        score, delta = self._rescore()

        task, advance = self._choose_task(extraction, slot)

        if advance and self.state.current_question_index >= len(prompts.QUESTION_PLAN):
            return self._finalise(score, delta)

        if advance:
            self.state.current_question_index += 1

        system = prompts.build_system_prompt(self.state.summary_for_prompt(), task)
        text = self.client.call(system, self._api_messages(), call_type="respond")
        self.state.add_turn("assistant", str(text))

        return TurnResult(text=str(text), score=score, delta=delta, label=confidence_label(score))

    # -------------------------------------------------------------- internals

    def _extract(self, user_text: str, slot: str) -> ProfileExtraction:
        question_index = self.state.current_question_index
        question = (
            prompts.QUESTION_PLAN[question_index - 1]
            if 1 <= question_index <= len(prompts.QUESTION_PLAN)
            else "(no question pending)"
        )
        system = prompts.build_extraction_prompt(
            self.state.summary_for_prompt(),
            f"Question {question_index} of 4, about the '{slot}' slot: {question}",
        )
        result = self.client.call(
            system,
            [{"role": "user", "content": user_text}],
            response_schema=ProfileExtraction,
            call_type="extract",
        )
        return result  # type: ignore[return-value]

    def _choose_task(self, extraction: ProfileExtraction, slot: str) -> tuple[str, bool]:
        """Pick this turn's instruction, and whether the interview advances.

        The one-follow-up rule lives here rather than in the prompt because a
        model asked to police its own question budget will not reliably do it.
        """
        used = self.state.clarifications_used.get(slot, 0)
        current = self.state.profile[slot]

        needs_clarifying = (
            slot not in self.state.declined_slots
            and (current.specificity in ("none", "vague") or current.is_ambiguous)
        )

        if extraction.off_topic_or_unintelligible and used < 1:
            self.state.clarifications_used[slot] = used + 1
            question = (
                prompts.QUESTION_PLAN[self.state.current_question_index - 1]
                if self.state.current_question_index >= 1
                else prompts.QUESTION_PLAN[0]
            )
            return prompts.TASK_OFF_TOPIC_V1.format(question=question), False

        if needs_clarifying and used < 1:
            self.state.clarifications_used[slot] = used + 1
            return prompts.TASK_CLARIFY_V1, False

        next_index = self.state.current_question_index + 1
        if next_index > len(prompts.QUESTION_PLAN):
            return "", True

        return (
            prompts.TASK_CONTINUE_V1.format(
                number=next_index, question=prompts.QUESTION_PLAN[next_index - 1]
            ),
            True,
        )

    def _finalise(self, score: float, delta: float) -> TurnResult:
        system = prompts.build_recommendation_prompt(
            self.state.summary_for_prompt(), self._transcript_text(), score
        )
        recommendation = self.client.call(
            system,
            self._api_messages(),
            response_schema=Recommendation,
            call_type="recommend",
        )
        assert isinstance(recommendation, Recommendation)

        self.state.recommendation = recommendation.model_dump()
        self.state.current_question_index = len(prompts.QUESTION_PLAN) + 1
        self.state.add_turn("assistant", recommendation.reasoning)

        return TurnResult(
            text=recommendation.reasoning,
            score=score,
            delta=delta,
            label=confidence_label(score),
            recommendation=recommendation,
            is_final=True,
        )

    @property
    def finished(self) -> bool:
        return self.state.recommendation is not None
