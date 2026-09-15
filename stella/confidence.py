"""Deterministic confidence scoring.

Pure functions over plain data. No LLM call happens here and none should: a
model asked "how confident are you, 0-100?" produces a number that cannot be
falsified, cannot be unit-tested, and drifts between turns for reasons nobody
can name. The model's only job is to label how *specific* each answer was;
the arithmetic is Python's, so the score moves for reasons we can point at.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .schemas import SLOT_NAMES, SlotExtraction

# Weights sum to 1.0. Rationale for the ordering lives in the README.
WEIGHTS: dict[str, float] = {
    "measurements": 0.35,
    "past_purchase": 0.30,
    "fit_preference": 0.20,
    "style_occasion": 0.15,
}

SPECIFICITY_SCORE: dict[str, float] = {
    "none": 0.0,
    "vague": 0.35,
    "partial": 0.7,
    "precise": 1.0,
}

# An answer that could resolve to two different sizes is worth keeping but is
# not worth full credit, so it is discounted rather than discarded.
AMBIGUITY_MULTIPLIER = 0.6

CONTRADICTION_PENALTY = 10.0

# A session missing a slot entirely should never present as near-certain,
# however precise the remaining answers were.
DECLINED_CEILING = 80.0

# A generic size label means different things across brands, so on its own it
# can never be more than half-credit evidence.
GENERIC_SIZE_CAP = 0.5

NARROWING_THRESHOLD = 40.0
CONFIDENT_THRESHOLD = 75.0

FULL_BLOCK = "█"
EMPTY_BLOCK = "░"


def slot_contribution(
    slot_name: str,
    slot: SlotExtraction,
    *,
    declined: bool = False,
    measurements_generic_only: bool = False,
) -> float:
    """Points this slot adds to the base score, before penalties."""
    if declined:
        return 0.0

    quality = SPECIFICITY_SCORE.get(slot.specificity, 0.0)

    if slot_name == "measurements" and measurements_generic_only:
        quality = min(quality, GENERIC_SIZE_CAP)

    if slot.is_ambiguous:
        quality *= AMBIGUITY_MULTIPLIER

    return 100.0 * WEIGHTS[slot_name] * quality


def compute_confidence(
    profile: Mapping[str, SlotExtraction],
    contradictions: Iterable | None = None,
    declined_slots: Iterable[str] | None = None,
    *,
    measurements_generic_only: bool = False,
) -> float:
    """Score the profile from 0 to 100.

    Monotonic in specificity: absent a new contradiction, a more specific
    answer can never lower the score. That property is what lets the agent
    narrate every movement honestly, and it is asserted in the tests.
    """
    declined = set(declined_slots or ())
    contradiction_count = len(list(contradictions or ()))

    base = sum(
        slot_contribution(
            name,
            profile.get(name, SlotExtraction()),
            declined=name in declined,
            measurements_generic_only=measurements_generic_only,
        )
        for name in SLOT_NAMES
    )

    score = base - CONTRADICTION_PENALTY * contradiction_count

    if declined:
        score = min(score, DECLINED_CEILING)

    return round(max(0.0, min(100.0, score)), 1)


def confidence_label(score: float) -> str:
    if score < NARROWING_THRESHOLD:
        return "Guessing"
    if score < CONFIDENT_THRESHOLD:
        return "Narrowing"
    return "Confident"


def render_bar(score: float, width: int = 30) -> str:
    """Fixed-width block bar. Always exactly `width` characters."""
    clamped = max(0.0, min(100.0, score))
    filled = int(round(clamped / 100.0 * width))
    filled = max(0, min(width, filled))
    return FULL_BLOCK * filled + EMPTY_BLOCK * (width - filled)


def breakdown(
    profile: Mapping[str, SlotExtraction],
    contradictions: Iterable | None = None,
    declined_slots: Iterable[str] | None = None,
    *,
    measurements_generic_only: bool = False,
) -> dict:
    """Per-slot arithmetic, for `/score` and for the README worked example."""
    declined = set(declined_slots or ())
    contradiction_list = list(contradictions or ())

    slots = []
    for name in SLOT_NAMES:
        slot = profile.get(name, SlotExtraction())
        is_declined = name in declined
        slots.append(
            {
                "slot": name,
                "weight": WEIGHTS[name],
                "specificity": "declined" if is_declined else slot.specificity,
                "ambiguous": slot.is_ambiguous and not is_declined,
                "points": round(
                    slot_contribution(
                        name,
                        slot,
                        declined=is_declined,
                        measurements_generic_only=measurements_generic_only,
                    ),
                    1,
                ),
            }
        )

    base = round(sum(s["points"] for s in slots), 1)
    final = compute_confidence(
        profile,
        contradiction_list,
        declined,
        measurements_generic_only=measurements_generic_only,
    )

    return {
        "slots": slots,
        "base": base,
        "contradiction_penalty": round(CONTRADICTION_PENALTY * len(contradiction_list), 1),
        "declined_ceiling_applied": bool(declined) and base - CONTRADICTION_PENALTY * len(
            contradiction_list
        ) > DECLINED_CEILING,
        "generic_size_cap_applied": measurements_generic_only,
        "final": final,
        "label": confidence_label(final),
    }
