"""Statistics and quality scoring utilities for PatchDuel evaluations."""
from __future__ import annotations

from .diff_engine import diff_stats


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English/code)."""
    return max(1, len(text) // 4)


def compute_quality_score(
    buggy_code: str,
    patch: str,
    expected_fix: str | None = None,
) -> int:
    """
    Return a quality score 0–100 for a generated patch.

    Scoring:
      100  — patch exactly matches the expected fix
      80–99 — small, targeted change (< 5% of file modified)
      65–79 — moderate change (5–15%)
      40–64 — large rewrite (15–40%)
      0–39  — no change made (model returned code as-is)
    """
    if not patch or not patch.strip():
        return 0

    # Exact match with expected fix wins immediately
    if expected_fix and patch.strip() == expected_fix.strip():
        return 100

    stats = diff_stats(buggy_code, patch)
    if not stats["changed"]:
        return 0

    total_changed = stats["insertions"] + stats["deletions"]
    total_original = max(1, len(buggy_code))
    change_ratio = total_changed / total_original

    if change_ratio < 0.05:
        score = 90
    elif change_ratio < 0.15:
        score = 80
    elif change_ratio < 0.40:
        score = 65
    else:
        score = 40

    # Bonus if patch approximates the expected fix
    if expected_fix:
        exp_stats = diff_stats(patch, expected_fix)
        if not exp_stats["changed"]:
            score = 100
        elif exp_stats["insertions"] + exp_stats["deletions"] < 10:
            score = min(100, score + 15)

    return score


def format_stats_line(
    dur_a: float,
    dur_b: float,
    tok_a: int,
    tok_b: int,
    model_a: str = "A",
    model_b: str = "B",
) -> str:
    """Human-readable stats line for display in the UI."""
    def _tok(n: int) -> str:
        return f"{n} tok" if n else "? tok"

    short_a = model_a.split("/")[-1][:20]
    short_b = model_b.split("/")[-1][:20]
    return (
        f"⏱ {short_a}: {dur_a:.1f}s | {_tok(tok_a)}    "
        f"⏱ {short_b}: {dur_b:.1f}s | {_tok(tok_b)}"
    )


def aggregate_session_stats(results: list[dict]) -> dict:
    """
    Compute aggregate statistics over a batch of results.

    Each result dict should contain: winner, duration_a, duration_b,
    tokens_a, tokens_b (all optional with graceful fallback).
    """
    total = len(results)
    wins_a = sum(1 for r in results if r.get("winner") == "Model A")
    wins_b = sum(1 for r in results if r.get("winner") == "Model B")
    ties = sum(1 for r in results if r.get("winner") == "Tie")
    both_wrong = sum(1 for r in results if r.get("winner") == "Both Wrong")

    durations_a = [r["duration_a"] for r in results if r.get("duration_a")]
    durations_b = [r["duration_b"] for r in results if r.get("duration_b")]
    tokens_a = [r["tokens_a"] for r in results if r.get("tokens_a")]
    tokens_b = [r["tokens_b"] for r in results if r.get("tokens_b")]

    return {
        "total": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "both_wrong": both_wrong,
        "win_rate_a": (wins_a / total) if total else 0.0,
        "win_rate_b": (wins_b / total) if total else 0.0,
        "avg_duration_a": (sum(durations_a) / len(durations_a)) if durations_a else 0.0,
        "avg_duration_b": (sum(durations_b) / len(durations_b)) if durations_b else 0.0,
        "total_tokens_a": sum(tokens_a),
        "total_tokens_b": sum(tokens_b),
    }


def heuristic_winner(
    buggy_code: str,
    patch_a: str,
    patch_b: str,
    expected_fix: str | None = None,
) -> str:
    """
    Determine a heuristic winner between two patches.

    Returns one of: "Model A", "Model B", "Tie", "Both Wrong".
    """
    if expected_fix:
        a_exact = patch_a.strip() == expected_fix.strip()
        b_exact = patch_b.strip() == expected_fix.strip()
        if a_exact and b_exact:
            return "Tie"
        if a_exact:
            return "Model A"
        if b_exact:
            return "Model B"

    stats_a = diff_stats(buggy_code, patch_a)
    stats_b = diff_stats(buggy_code, patch_b)
    changed_a = stats_a["insertions"] + stats_a["deletions"]
    changed_b = stats_b["insertions"] + stats_b["deletions"]

    # Check if both models made no changes first (before checking if patches are equal)
    if changed_a == 0 and changed_b == 0:
        return "Both Wrong"
    if patch_a.strip() == patch_b.strip():
        return "Tie"
    if changed_a == 0:
        return "Model B"
    if changed_b == 0:
        return "Model A"
    # Fewer chars changed = more surgical fix
    return "Model A" if changed_a <= changed_b else "Model B"
