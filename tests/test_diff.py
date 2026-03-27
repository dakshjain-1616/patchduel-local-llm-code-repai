"""Tests for the diff engine."""
import pytest
from patchduel_local_llm_.diff_engine import (
    compute_char_diff,
    compute_line_diff,
    diff_stats,
    diff_summary,
    render_html_diff,
)

# ── Test spec case 1 ───────────────────────────────────────────────────────────
BUGGY_ADD = "def add(a,b): return a-b"
FIXED_ADD  = "def add(a,b): return a+b"

BUGGY_MULTILINE = "def add(a, b):\n    return a - b\n"
FIXED_MULTILINE = "def add(a, b):\n    return a + b\n"


class TestComputeDiff:
    def test_identical_returns_equal_ops_only(self):
        diffs = compute_char_diff("hello", "hello")
        ops = [op for op, _ in diffs]
        assert all(op == 0 for op in ops), "Expected only EQUAL ops for identical strings"

    def test_single_char_change_detected(self):
        diffs = compute_char_diff(BUGGY_ADD, FIXED_ADD)
        ops = {op for op, _ in diffs}
        # Must have at least one insertion and one deletion
        assert 1 in ops, "Expected insertion op"
        assert -1 in ops, "Expected deletion op"

    def test_changed_char_is_operator(self):
        diffs = compute_char_diff(BUGGY_ADD, FIXED_ADD)
        deleted = "".join(t for op, t in diffs if op == -1)
        inserted = "".join(t for op, t in diffs if op == 1)
        assert "-" in deleted, f"Expected '-' in deleted chars, got: {deleted!r}"
        assert "+" in inserted, f"Expected '+' in inserted chars, got: {inserted!r}"

    def test_empty_strings(self):
        diffs = compute_char_diff("", "")
        assert diffs == [] or all(op == 0 for op, _ in diffs)

    def test_insertion_only(self):
        diffs = compute_char_diff("ab", "abc")
        inserted = "".join(t for op, t in diffs if op == 1)
        assert "c" in inserted

    def test_deletion_only(self):
        diffs = compute_char_diff("abc", "ab")
        deleted = "".join(t for op, t in diffs if op == -1)
        assert "c" in deleted

    def test_line_diff_detects_return_line(self):
        diffs = compute_line_diff(BUGGY_MULTILINE, FIXED_MULTILINE)
        deleted = "".join(t for op, t in diffs if op == -1)
        inserted = "".join(t for op, t in diffs if op == 1)
        assert "a - b" in deleted or "-" in deleted
        assert "a + b" in inserted or "+" in inserted


class TestDiffStats:
    def test_stats_no_change(self):
        s = diff_stats("hello", "hello")
        assert s["insertions"] == 0
        assert s["deletions"] == 0
        assert s["changed"] is False

    def test_stats_single_char_change(self):
        s = diff_stats(BUGGY_ADD, FIXED_ADD)
        assert s["insertions"] >= 1
        assert s["deletions"] >= 1
        assert s["changed"] is True

    def test_stats_pure_insertion(self):
        s = diff_stats("ab", "abc")
        assert s["insertions"] >= 1
        assert s["deletions"] == 0

    def test_stats_pure_deletion(self):
        s = diff_stats("abc", "ab")
        assert s["insertions"] == 0
        assert s["deletions"] >= 1

    def test_summary_no_change(self):
        s = diff_summary("same", "same")
        assert s == "No changes"

    def test_summary_shows_counts(self):
        s = diff_summary(BUGGY_ADD, FIXED_ADD)
        assert "+" in s or "added" in s or "removed" in s


class TestRenderHtmlDiff:
    def test_returns_string(self):
        html = render_html_diff(BUGGY_ADD, FIXED_ADD)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_diff_markers(self):
        html = render_html_diff(BUGGY_ADD, FIXED_ADD)
        # Should contain both a deletion (red) and insertion (green) class
        assert "diff-add" in html or "#3fb950" in html
        assert "diff-del" in html or "#f85149" in html

    def test_html_is_valid_fragment(self):
        html = render_html_diff(BUGGY_ADD, FIXED_ADD)
        assert "<div" in html
        assert "</div>" in html

    def test_multiline_diff_highlights_return_line(self):
        """Test spec case 1: the return line diff is visible in the HTML."""
        html = render_html_diff(BUGGY_MULTILINE, FIXED_MULTILINE)
        # The deleted line should contain 'a - b'
        assert "a - b" in html or "a&nbsp;-&nbsp;b" in html or "a &#45; b" in html or "-" in html
        # The inserted line should contain 'a + b'
        assert "a + b" in html or "a&nbsp;+&nbsp;b" in html or "+" in html

    def test_no_diff_shows_no_changes_message(self):
        html = render_html_diff("x = 1", "x = 1")
        assert "no changes" in html.lower() or "diff-eq" in html

    def test_complete_replacement(self):
        html = render_html_diff("foo()", "bar()")
        assert "diff-add" in html or "diff-del" in html
