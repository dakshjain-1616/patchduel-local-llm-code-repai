"""Tests for the scenario library."""
import pytest

from patchduel_local_llm_.scenarios import (
    SCENARIOS,
    get_all_scenarios,
    get_scenario_by_id,
    get_scenarios_by_tag,
    mock_patch,
    scenario_choices,
)


class TestGetAllScenarios:
    def test_returns_list(self):
        assert isinstance(get_all_scenarios(), list)

    def test_at_least_five_scenarios(self):
        assert len(get_all_scenarios()) >= 5

    def test_each_scenario_has_required_keys(self):
        required = {"id", "name", "description", "tags", "language", "buggy_code", "expected_fix"}
        for s in get_all_scenarios():
            assert required.issubset(s.keys()), f"{s['id']} missing keys"

    def test_ids_are_unique(self):
        ids = [s["id"] for s in get_all_scenarios()]
        assert len(ids) == len(set(ids))

    def test_buggy_and_expected_are_different(self):
        for s in get_all_scenarios():
            assert s["buggy_code"].strip() != s["expected_fix"].strip(), (
                f"{s['id']}: buggy_code and expected_fix must differ"
            )


class TestGetScenarioById:
    def test_known_id_returns_dict(self):
        result = get_scenario_by_id("sc-001")
        assert result is not None
        assert result["id"] == "sc-001"

    def test_unknown_id_returns_none(self):
        assert get_scenario_by_id("sc-9999") is None

    def test_all_ids_findable(self):
        for s in SCENARIOS:
            found = get_scenario_by_id(s["id"])
            assert found is not None
            assert found["id"] == s["id"]


class TestGetScenariosByTag:
    def test_easy_tag_returns_subset(self):
        easy = get_scenarios_by_tag("easy")
        assert isinstance(easy, list)
        assert len(easy) >= 1
        assert all("easy" in s["tags"] for s in easy)

    def test_nonexistent_tag_returns_empty(self):
        assert get_scenarios_by_tag("__no_such_tag__") == []

    def test_medium_tag_works(self):
        medium = get_scenarios_by_tag("medium")
        assert len(medium) >= 1


class TestScenarioChoices:
    def test_returns_list_of_tuples(self):
        choices = scenario_choices()
        assert isinstance(choices, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in choices)

    def test_first_choice_is_custom(self):
        choices = scenario_choices()
        assert choices[0][1] == "custom"

    def test_all_scenario_ids_present(self):
        ids = {c[1] for c in scenario_choices()}
        for s in SCENARIOS:
            assert s["id"] in ids


class TestMockPatch:
    def test_known_buggy_code_returns_expected_fix(self):
        for s in SCENARIOS:
            result = mock_patch(s["buggy_code"])
            assert result == s["expected_fix"], (
                f"{s['id']}: mock_patch should return expected_fix for known code"
            )

    def test_unknown_code_returns_input_unchanged(self):
        code = "x = totally_unknown_code()"
        result = mock_patch(code)
        assert result == code

    def test_whitespace_stripped_for_matching(self):
        s = SCENARIOS[0]
        # Add leading/trailing newline — should still match
        result = mock_patch("\n" + s["buggy_code"] + "\n")
        assert result == s["expected_fix"]
