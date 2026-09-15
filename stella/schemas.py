"""Typed boundaries between the LLM and the rest of the program.

Nothing crosses from model output into application state without passing
through one of these models first.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Specificity = Literal["none", "vague", "partial", "precise"]

SlotName = Literal["measurements", "fit_preference", "style_occasion", "past_purchase"]

SLOT_NAMES: tuple[str, ...] = (
    "measurements",
    "fit_preference",
    "style_occasion",
    "past_purchase",
)


class SlotExtraction(BaseModel):
    """One slot's worth of signal, as read out of a single user reply."""

    value: str | None = Field(
        default=None,
        description="Verbatim or lightly normalised content the user supplied. "
        "None when the user said nothing about this slot.",
    )
    specificity: Specificity = Field(
        default="none",
        description="How actionable the value is for choosing a size. "
        "Not a measure of how many words the user wrote.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True when the value could plausibly resolve to two different sizes.",
    )
    contradicts_previous: bool = Field(
        default=False,
        description="True when this value conflicts with what the profile already holds.",
    )
    note: str | None = Field(
        default=None,
        description="Short analyst note: what was unclear, or what the contradiction was.",
    )
    declined: bool = Field(
        default=False,
        description="True when the user explicitly refused or skipped this slot "
        "(e.g. 'skip', 'rather not say'). Distinct from simply not mentioning it.",
    )


class ProfileExtraction(BaseModel):
    """The full read of one user turn across all four slots."""

    measurements: SlotExtraction = Field(default_factory=SlotExtraction)
    fit_preference: SlotExtraction = Field(default_factory=SlotExtraction)
    style_occasion: SlotExtraction = Field(default_factory=SlotExtraction)
    past_purchase: SlotExtraction = Field(default_factory=SlotExtraction)

    measurements_are_generic_sizes_only: bool = Field(
        default=False,
        description="True when the only sizing information is a generic label "
        "(S/M/L, 'a medium') with no brand, garment or body measurement to anchor it. "
        "Generic labels mean different things across brands, so Python caps their "
        "contribution to the score.",
    )
    off_topic_or_unintelligible: bool = Field(
        default=False,
        description="True when the reply carries no fit signal at all — gibberish, "
        "an unrelated request, or pure chit-chat.",
    )

    def slots(self) -> dict[str, SlotExtraction]:
        return {name: getattr(self, name) for name in SLOT_NAMES}


class Recommendation(BaseModel):
    """The terminal output of a session."""

    size_range: str = Field(description="e.g. 'US M-L, or 40-42 EU in structured brands'.")
    silhouette: str = Field(description="Cut and shape guidance driven by fit preference.")
    brand_tip: str = Field(description="How to adjust across brands that run small or large.")
    reasoning: str = Field(
        description="Which collected slots drove which part of the answer. "
        "Must name the slots explicitly."
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Hedges scaled to the final confidence. At low confidence this "
        "must include what additional information would raise the score.",
    )
