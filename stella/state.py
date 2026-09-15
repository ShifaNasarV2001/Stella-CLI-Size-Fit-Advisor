"""The single source of truth for a session.

The model never holds state. Everything it is allowed to "remember" is
serialised out of this object and injected into the prompt on every turn,
which is what makes the session inspectable and reproducible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .schemas import SLOT_NAMES, ProfileExtraction, SlotExtraction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Turn:
    role: str
    content: str
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, raw: dict) -> "Turn":
        return cls(role=raw["role"], content=raw["content"], timestamp=raw["timestamp"])


@dataclass
class Contradiction:
    slot: str
    previous_value: str
    new_value: str
    note: str | None = None
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "note": self.note,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Contradiction":
        return cls(**raw)

    def describe(self) -> str:
        return (
            f"{self.slot}: earlier '{self.previous_value}' vs now '{self.new_value}'"
            + (f" ({self.note})" if self.note else "")
        )


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=_now)
    transcript: list[Turn] = field(default_factory=list)
    profile: dict[str, SlotExtraction] = field(
        default_factory=lambda: {name: SlotExtraction() for name in SLOT_NAMES}
    )
    confidence: float = 0.0
    confidence_history: list[float] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    declined_slots: set[str] = field(default_factory=set)
    current_question_index: int = 0
    measurements_generic_only: bool = False
    clarifications_used: dict[str, int] = field(default_factory=dict)
    recommendation: dict | None = None

    # ---------------------------------------------------------------- mutation

    def add_turn(self, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        self.transcript.append(turn)
        return turn

    def merge_extraction(self, extraction: ProfileExtraction) -> list[Contradiction]:
        """Fold one turn's extraction into the profile.

        A later turn never silently erases an earlier one. A slot is only
        touched when the new reading actually carries signal, and a conflicting
        reading is recorded as a Contradiction before the newer value wins.
        """
        newly_recorded: list[Contradiction] = []

        for slot, incoming in extraction.slots().items():
            existing = self.profile[slot]

            if incoming.declined:
                self.declined_slots.add(slot)
                self.profile[slot] = SlotExtraction(
                    value=existing.value,
                    specificity=existing.specificity,
                    is_ambiguous=existing.is_ambiguous,
                    declined=True,
                    note=incoming.note or "User declined to answer.",
                )
                continue

            carries_signal = incoming.value is not None or incoming.specificity != "none"
            if not carries_signal:
                continue

            if (
                incoming.contradicts_previous
                and existing.value
                and incoming.value
                and existing.value.strip().lower() != incoming.value.strip().lower()
            ):
                record = Contradiction(
                    slot=slot,
                    previous_value=existing.value,
                    new_value=incoming.value,
                    note=incoming.note,
                )
                self.contradictions.append(record)
                newly_recorded.append(record)

            # The user has now answered, so an earlier decline no longer applies.
            self.declined_slots.discard(slot)
            self.profile[slot] = incoming

        if extraction.measurements_are_generic_sizes_only:
            self.measurements_generic_only = True
        elif extraction.measurements.specificity in ("partial", "precise"):
            self.measurements_generic_only = False

        return newly_recorded

    def record_confidence(self, score: float) -> float:
        """Store a freshly computed score and return the delta from the last one."""
        previous = self.confidence
        self.confidence = score
        self.confidence_history.append(score)
        return round(score - previous, 1)

    # -------------------------------------------------------------- inspection

    def answered_slots(self) -> list[str]:
        return [
            name
            for name, slot in self.profile.items()
            if slot.specificity != "none" and name not in self.declined_slots
        ]

    def summary_for_prompt(self) -> str:
        """The compact state block injected into every system prompt.

        Deliberately terse: the model reads this many times per session, and a
        verbose dump invites it to paraphrase rather than respect the values.
        """
        lines = [
            f"session_id: {self.session_id}",
            f"questions_asked: {self.current_question_index} of 4",
            f"confidence: {self.confidence:.1f}/100",
        ]

        if len(self.confidence_history) >= 2:
            delta = self.confidence_history[-1] - self.confidence_history[-2]
            lines.append(f"confidence_change_this_turn: {delta:+.1f}")

        lines.append("profile:")
        for name in SLOT_NAMES:
            slot = self.profile[name]
            if name in self.declined_slots:
                lines.append(f"  {name}: DECLINED BY USER — do not ask again")
                continue
            if slot.specificity == "none" and not slot.value:
                lines.append(f"  {name}: (not yet collected)")
                continue
            flags = []
            if slot.is_ambiguous:
                flags.append("AMBIGUOUS")
            if slot.note:
                flags.append(f"note={slot.note}")
            suffix = f" [{'; '.join(flags)}]" if flags else ""
            lines.append(f"  {name}: {slot.value!r} (specificity={slot.specificity}){suffix}")

        if self.measurements_generic_only:
            lines.append(
                "  ^ sizing is a generic label only (no brand or body measurement to anchor it)"
            )

        if self.contradictions:
            lines.append("unresolved_contradictions:")
            for c in self.contradictions:
                lines.append(f"  - {c.describe()}")
        else:
            lines.append("unresolved_contradictions: none")

        return "\n".join(lines)

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "transcript": [t.to_dict() for t in self.transcript],
            "profile": {k: v.model_dump() for k, v in self.profile.items()},
            "confidence": self.confidence,
            "confidence_history": self.confidence_history,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "declined_slots": sorted(self.declined_slots),
            "current_question_index": self.current_question_index,
            "measurements_generic_only": self.measurements_generic_only,
            "clarifications_used": self.clarifications_used,
            "recommendation": self.recommendation,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, raw: dict) -> "SessionState":
        return cls(
            session_id=raw["session_id"],
            created_at=raw["created_at"],
            transcript=[Turn.from_dict(t) for t in raw["transcript"]],
            profile={k: SlotExtraction(**v) for k, v in raw["profile"].items()},
            confidence=raw["confidence"],
            confidence_history=list(raw["confidence_history"]),
            contradictions=[Contradiction.from_dict(c) for c in raw["contradictions"]],
            declined_slots=set(raw["declined_slots"]),
            current_question_index=raw["current_question_index"],
            measurements_generic_only=raw.get("measurements_generic_only", False),
            clarifications_used=dict(raw.get("clarifications_used", {})),
            recommendation=raw.get("recommendation"),
        )

    @classmethod
    def from_json(cls, blob: str) -> "SessionState":
        return cls.from_dict(json.loads(blob))
