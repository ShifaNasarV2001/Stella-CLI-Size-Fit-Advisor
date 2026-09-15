import itertools

import pytest

from stella.confidence import (
    CONFIDENT_THRESHOLD,
    CONTRADICTION_PENALTY,
    DECLINED_CEILING,
    NARROWING_THRESHOLD,
    SPECIFICITY_SCORE,
    WEIGHTS,
    compute_confidence,
    confidence_label,
    breakdown,
    render_bar,
)
from stella.schemas import SLOT_NAMES, SlotExtraction

SPECIFICITY_ORDER = ["none", "vague", "partial", "precise"]


def profile(**kwargs) -> dict[str, SlotExtraction]:
    """Build a profile; kwargs are slot -> specificity string or SlotExtraction."""
    out = {name: SlotExtraction() for name in SLOT_NAMES}
    for name, spec in kwargs.items():
        if isinstance(spec, SlotExtraction):
            out[name] = spec
        else:
            out[name] = SlotExtraction(value=f"{name} answer", specificity=spec)
    return out


def test_weights_sum_to_one():
    assert pytest.approx(sum(WEIGHTS.values()), abs=1e-9) == 1.0
    assert set(WEIGHTS) == set(SLOT_NAMES)


def test_empty_profile_scores_zero():
    assert compute_confidence(profile()) == 0.0


def test_fully_precise_profile_scores_one_hundred():
    assert compute_confidence(profile(**{n: "precise" for n in SLOT_NAMES})) == 100.0


def test_monotonic_in_specificity_for_every_slot():
    """More specific input never lowers the score, absent a contradiction."""
    for slot in SLOT_NAMES:
        scores = [compute_confidence(profile(**{slot: s})) for s in SPECIFICITY_ORDER]
        assert scores == sorted(scores), f"{slot} is not monotonic: {scores}"
        assert scores[0] < scores[-1]


def test_monotonic_across_combinations():
    """Upgrading any one slot in an arbitrary profile never reduces the total."""
    for combo in itertools.product(SPECIFICITY_ORDER, repeat=len(SLOT_NAMES)):
        base_profile = dict(zip(SLOT_NAMES, combo))
        base_score = compute_confidence(profile(**base_profile))
        for slot, current in base_profile.items():
            idx = SPECIFICITY_ORDER.index(current)
            if idx == len(SPECIFICITY_ORDER) - 1:
                continue
            upgraded = dict(base_profile)
            upgraded[slot] = SPECIFICITY_ORDER[idx + 1]
            assert compute_confidence(profile(**upgraded)) >= base_score


def test_slot_weighting_matches_declared_priority():
    """A precise answer in a heavier slot is worth more than in a lighter one."""
    ranked = sorted(WEIGHTS, key=WEIGHTS.get, reverse=True)
    scores = [compute_confidence(profile(**{slot: "precise"})) for slot in ranked]
    assert scores == sorted(scores, reverse=True)


def test_ambiguity_discounts_but_does_not_erase():
    plain = compute_confidence(profile(measurements="precise"))
    ambiguous = compute_confidence(
        profile(
            measurements=SlotExtraction(
                value="medium-ish", specificity="precise", is_ambiguous=True
            )
        )
    )
    assert 0 < ambiguous < plain


def test_contradiction_penalty_is_ten_per_contradiction():
    p = profile(**{n: "precise" for n in SLOT_NAMES})
    clean = compute_confidence(p, [], set())
    one = compute_confidence(p, ["c1"], set())
    two = compute_confidence(p, ["c1", "c2"], set())
    assert clean - one == pytest.approx(CONTRADICTION_PENALTY)
    assert one - two == pytest.approx(CONTRADICTION_PENALTY)


def test_declined_slot_contributes_zero_and_caps_ceiling():
    p = profile(**{n: "precise" for n in SLOT_NAMES})
    score = compute_confidence(p, [], {"style_occasion"})
    # style_occasion is worth 15 points, so the raw total would be 85;
    # the declined ceiling pulls it down to 80.
    assert score == DECLINED_CEILING


def test_declined_ceiling_does_not_inflate_a_low_score():
    p = profile(measurements="vague")
    score = compute_confidence(p, [], {"past_purchase"})
    assert score < DECLINED_CEILING
    assert score == pytest.approx(
        round(100 * WEIGHTS["measurements"] * SPECIFICITY_SCORE["vague"], 1)
    )


def test_generic_sizes_cap_the_measurement_slot():
    p = profile(measurements="precise")
    uncapped = compute_confidence(p, measurements_generic_only=False)
    capped = compute_confidence(p, measurements_generic_only=True)
    assert capped == pytest.approx(uncapped * 0.5)


def test_generic_cap_never_raises_a_low_specificity_slot():
    p = profile(measurements="vague")  # 0.35 quality, below the 0.5 cap
    assert compute_confidence(p, measurements_generic_only=True) == compute_confidence(
        p, measurements_generic_only=False
    )


def test_score_is_clamped_to_zero():
    p = profile(measurements="vague")
    assert compute_confidence(p, ["c"] * 50, set()) == 0.0


def test_score_is_clamped_to_one_hundred():
    p = profile(**{n: "precise" for n in SLOT_NAMES})
    assert compute_confidence(p) <= 100.0


def test_labels_partition_the_range():
    assert confidence_label(0) == "Guessing"
    assert confidence_label(NARROWING_THRESHOLD - 0.1) == "Guessing"
    assert confidence_label(NARROWING_THRESHOLD) == "Narrowing"
    assert confidence_label(CONFIDENT_THRESHOLD - 0.1) == "Narrowing"
    assert confidence_label(CONFIDENT_THRESHOLD) == "Confident"
    assert confidence_label(100) == "Confident"


@pytest.mark.parametrize("score", [-50, 0, 0.4, 33.3, 50, 99.9, 100, 150])
@pytest.mark.parametrize("width", [10, 30, 47])
def test_bar_is_always_exactly_width_characters(score, width):
    bar = render_bar(score, width=width)
    assert len(bar) == width
    assert set(bar) <= {"█", "░"}


def test_bar_fills_monotonically():
    widths = [render_bar(s).count("█") for s in range(0, 101, 5)]
    assert widths == sorted(widths)
    assert render_bar(0).count("█") == 0
    assert render_bar(100).count("░") == 0


def test_breakdown_arithmetic_reconciles_with_compute_confidence():
    p = profile(measurements="precise", fit_preference="vague")
    detail = breakdown(p, ["c1"], set())
    assert detail["base"] == pytest.approx(sum(s["points"] for s in detail["slots"]))
    assert detail["contradiction_penalty"] == CONTRADICTION_PENALTY
    assert detail["final"] == compute_confidence(p, ["c1"], set())
    assert detail["label"] == confidence_label(detail["final"])
