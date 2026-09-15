from stella.schemas import ProfileExtraction, SlotExtraction
from stella.state import SessionState


def extraction(**slots) -> ProfileExtraction:
    return ProfileExtraction(**slots)


def test_add_turn_appends_in_order():
    state = SessionState()
    state.add_turn("assistant", "first")
    state.add_turn("user", "second")
    assert [t.role for t in state.transcript] == ["assistant", "user"]
    assert [t.content for t in state.transcript] == ["first", "second"]
    assert all(t.timestamp for t in state.transcript)


def test_merge_populates_profile():
    state = SessionState()
    state.merge_extraction(
        extraction(measurements=SlotExtraction(value="5'6\", 140lb", specificity="precise"))
    )
    assert state.profile["measurements"].value == "5'6\", 140lb"
    assert state.profile["measurements"].specificity == "precise"
    assert state.answered_slots() == ["measurements"]


def test_merge_ignores_slots_with_no_signal():
    state = SessionState()
    state.merge_extraction(
        extraction(fit_preference=SlotExtraction(value="relaxed", specificity="precise"))
    )
    state.merge_extraction(extraction(measurements=SlotExtraction()))  # empty reading
    assert state.profile["fit_preference"].value == "relaxed"
    assert state.profile["measurements"].value is None


def test_merge_does_not_silently_overwrite_a_contradiction():
    state = SessionState()
    state.merge_extraction(
        extraction(measurements=SlotExtraction(value="size 8", specificity="partial"))
    )
    recorded = state.merge_extraction(
        extraction(
            measurements=SlotExtraction(
                value="size 12",
                specificity="partial",
                contradicts_previous=True,
                note="said 8 earlier",
            )
        )
    )
    assert len(recorded) == 1
    assert len(state.contradictions) == 1
    c = state.contradictions[0]
    assert c.slot == "measurements"
    assert c.previous_value == "size 8"
    assert c.new_value == "size 12"
    # The newer value becomes authoritative, but the conflict survives in state.
    assert state.profile["measurements"].value == "size 12"
    assert "size 8" in state.summary_for_prompt()


def test_restating_the_same_value_is_not_a_contradiction():
    state = SessionState()
    state.merge_extraction(
        extraction(measurements=SlotExtraction(value="Size 8", specificity="partial"))
    )
    state.merge_extraction(
        extraction(
            measurements=SlotExtraction(
                value="size 8", specificity="partial", contradicts_previous=True
            )
        )
    )
    assert state.contradictions == []


def test_decline_is_recorded_and_reversible():
    state = SessionState()
    state.merge_extraction(
        extraction(past_purchase=SlotExtraction(declined=True, note="rather not say"))
    )
    assert "past_purchase" in state.declined_slots
    assert "DECLINED" in state.summary_for_prompt()

    state.merge_extraction(
        extraction(past_purchase=SlotExtraction(value="Uniqlo M", specificity="precise"))
    )
    assert "past_purchase" not in state.declined_slots
    assert state.profile["past_purchase"].value == "Uniqlo M"


def test_generic_size_flag_sets_and_clears():
    state = SessionState()
    state.merge_extraction(
        extraction(
            measurements=SlotExtraction(value="medium", specificity="vague"),
            measurements_are_generic_sizes_only=True,
        )
    )
    assert state.measurements_generic_only is True

    state.merge_extraction(
        extraction(measurements=SlotExtraction(value="38in chest", specificity="precise"))
    )
    assert state.measurements_generic_only is False


def test_record_confidence_tracks_history_and_delta():
    state = SessionState()
    assert state.record_confidence(20.0) == 20.0
    assert state.record_confidence(52.5) == 32.5
    assert state.record_confidence(42.5) == -10.0
    assert state.confidence_history == [20.0, 52.5, 42.5]
    assert state.confidence == 42.5


def test_summary_marks_uncollected_slots():
    summary = SessionState().summary_for_prompt()
    assert summary.count("(not yet collected)") == 4
    assert "unresolved_contradictions: none" in summary


def test_json_round_trip_preserves_everything():
    state = SessionState()
    state.add_turn("assistant", "What are your measurements?")
    state.add_turn("user", "About a medium")
    state.merge_extraction(
        extraction(
            measurements=SlotExtraction(
                value="medium", specificity="vague", is_ambiguous=True, note="brand unknown"
            ),
            measurements_are_generic_sizes_only=True,
        )
    )
    state.merge_extraction(
        extraction(
            measurements=SlotExtraction(
                value="large", specificity="vague", contradicts_previous=True
            ),
            style_occasion=SlotExtraction(declined=True),
        )
    )
    state.record_confidence(17.5)
    state.current_question_index = 2
    state.recommendation = {"size_range": "M-L"}

    restored = SessionState.from_json(state.to_json())

    assert restored.to_dict() == state.to_dict()
    assert restored.declined_slots == state.declined_slots
    assert restored.profile["measurements"].value == "large"
    assert restored.measurements_generic_only is True
    assert restored.confidence_history == [17.5]
    assert len(restored.contradictions) == 1
    assert restored.contradictions[0].previous_value == "medium"
    assert restored.summary_for_prompt() == state.summary_for_prompt()


def test_to_json_is_valid_json_and_indented():
    import json

    blob = SessionState().to_json()
    assert json.loads(blob)["confidence"] == 0.0
    assert "\n  " in blob
