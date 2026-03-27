"""Tests for the stats/quality scoring module."""
import pytest

from patchduel_local_llm_.stats import (
    aggregate_session_stats,
    compute_quality_score,
    estimate_tokens,
    format_stats_line,
    heuristic_winner,
)


class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert estimate_tokens("") == 1

    def test_short_text(self):
        # "hello" = 5 chars → 5//4 = 1, but max(1, 1) = 1
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "x" * 100
        assert estimate_tokens(text) == 25

    def test_returns_int(self):
        assert isinstance(estimate_tokens("some text here"), int)


class TestComputeQualityScore:
    def test_empty_patch_returns_zero(self):
        assert compute_quality_score("def foo(): pass", "") == 0

    def test_whitespace_only_patch_returns_zero(self):
        assert compute_quality_score("def foo(): pass", "   ") == 0

    def test_unchanged_patch_returns_zero(self):
        code = "def foo():\n    return 1"
        assert compute_quality_score(code, code) == 0

    def test_exact_match_with_expected_returns_100(self):
        buggy = "def add(a, b):\n    return a - b"
        expected = "def add(a, b):\n    return a + b"
        score = compute_quality_score(buggy, expected, expected_fix=expected)
        assert score == 100

    def test_small_change_scores_high(self):
        # Change 1 char in a 30-char string → ~3% change → score 90
        buggy = "def add(a, b):\n    return a - b"
        patch = "def add(a, b):\n    return a + b"
        score = compute_quality_score(buggy, patch)
        assert score >= 80

    def test_large_rewrite_scores_lower(self):
        buggy = "x = 1"
        # Replace entirely with something much longer
        patch = "x = 1\n" + "# comment\n" * 20
        score = compute_quality_score(buggy, patch)
        assert score < 80

    def test_returns_int(self):
        score = compute_quality_score("def f(): pass", "def f(): return 1")
        assert isinstance(score, int)

    def test_score_in_valid_range(self):
        score = compute_quality_score("a = 1 + 1", "a = 2")
        assert 0 <= score <= 100


class TestHeuristicWinner:
    def test_both_unchanged_returns_both_wrong(self):
        code = "def f(): pass"
        assert heuristic_winner(code, code, code) == "Both Wrong"

    def test_identical_patches_returns_tie(self):
        buggy = "x = 1 - 1"
        patch = "x = 1 + 1"
        assert heuristic_winner(buggy, patch, patch) == "Tie"

    def test_only_a_changes_returns_model_b_wrong(self):
        buggy = "x = 1 - 1"
        patch_a = "x = 1 + 1"
        # model B returns same as buggy
        result = heuristic_winner(buggy, patch_a, buggy)
        assert result == "Model A"

    def test_only_b_changes_returns_model_a_wrong(self):
        buggy = "x = 1 - 1"
        patch_b = "x = 1 + 1"
        result = heuristic_winner(buggy, buggy, patch_b)
        assert result == "Model B"

    def test_exact_expected_match_wins(self):
        buggy = "def add(a, b):\n    return a - b"
        expected = "def add(a, b):\n    return a + b"
        large_patch = "def add(a, b):\n    # Fixed\n    return a + b"
        # Model A matches exactly, Model B has extra comment
        result = heuristic_winner(buggy, expected, large_patch, expected_fix=expected)
        assert result == "Model A"

    def test_both_match_expected_returns_tie(self):
        buggy = "x = 1 - 1"
        expected = "x = 1 + 1"
        result = heuristic_winner(buggy, expected, expected, expected_fix=expected)
        assert result == "Tie"


class TestAggregateSessionStats:
    def _make_results(self, winners):
        return [
            {"winner": w, "duration_a": 1.0, "duration_b": 2.0, "tokens_a": 100, "tokens_b": 150}
            for w in winners
        ]

    def test_empty_results(self):
        stats = aggregate_session_stats([])
        assert stats["total"] == 0
        assert stats["win_rate_a"] == 0.0

    def test_all_model_a_wins(self):
        stats = aggregate_session_stats(self._make_results(["Model A"] * 5))
        assert stats["wins_a"] == 5
        assert stats["wins_b"] == 0
        assert stats["win_rate_a"] == 1.0

    def test_mixed_results(self):
        stats = aggregate_session_stats(
            self._make_results(["Model A", "Model B", "Tie", "Both Wrong", "Model A"])
        )
        assert stats["total"] == 5
        assert stats["wins_a"] == 2
        assert stats["wins_b"] == 1
        assert stats["ties"] == 1
        assert stats["both_wrong"] == 1

    def test_avg_duration_computed(self):
        stats = aggregate_session_stats(self._make_results(["Model A", "Model B"]))
        assert stats["avg_duration_a"] == 1.0
        assert stats["avg_duration_b"] == 2.0

    def test_total_tokens_summed(self):
        stats = aggregate_session_stats(self._make_results(["Tie"] * 3))
        assert stats["total_tokens_a"] == 300
        assert stats["total_tokens_b"] == 450


class TestFormatStatsLine:
    def test_returns_string(self):
        result = format_stats_line(1.5, 2.3, 120, 200)
        assert isinstance(result, str)

    def test_contains_durations(self):
        result = format_stats_line(1.5, 2.3, 120, 200)
        assert "1.5s" in result
        assert "2.3s" in result

    def test_contains_tokens(self):
        result = format_stats_line(1.5, 2.3, 120, 200)
        assert "120 tok" in result
        assert "200 tok" in result

    def test_zero_tokens_shows_question_mark(self):
        result = format_stats_line(1.0, 2.0, 0, 0)
        assert "? tok" in result

    def test_custom_model_names(self):
        result = format_stats_line(1.0, 2.0, 50, 60, model_a="llama3", model_b="mistral")
        assert "llama3" in result
        assert "mistral" in result
