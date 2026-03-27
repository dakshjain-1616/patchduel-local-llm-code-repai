"""
02_advanced_usage.py — Advanced PatchDuel features.

Demonstrates:
- Browsing the scenario library by difficulty tag
- Computing quality scores for patches
- Heuristic winner determination between two patches
- Persisting results to a temporary SQLite database
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

from patchduel_local_llm_.scenarios import get_scenarios_by_tag, mock_patch
from patchduel_local_llm_.stats import compute_quality_score, heuristic_winner
from patchduel_local_llm_.database import init_db, save_run, get_runs, run_count

# ── 1. Browse scenarios by difficulty ─────────────────────────────────────────
print("=== Easy scenarios ===")
easy = get_scenarios_by_tag("easy")
for s in easy:
    print(f"  [{s['id']}] {s['name']} — {s['description']}")
print()

# ── 2. Score a patch against the expected fix ─────────────────────────────────
from patchduel_local_llm_.scenarios import get_scenario_by_id

sc = get_scenario_by_id("sc-001")
buggy = sc["buggy_code"]
expected = sc["expected_fix"]

# Perfect fix
score_perfect = compute_quality_score(buggy, expected, expected_fix=expected)
# Small but wrong change
small_wrong = buggy.replace("a - b", "a * b")
score_wrong = compute_quality_score(buggy, small_wrong, expected_fix=expected)
# Unchanged
score_none = compute_quality_score(buggy, buggy, expected_fix=expected)

print("=== Quality scores ===")
print(f"  Perfect fix:    {score_perfect}/100")
print(f"  Small wrong fix:{score_wrong}/100")
print(f"  No change:      {score_none}/100")
print()

# ── 3. Heuristic winner between two patches ───────────────────────────────────
patch_a = mock_patch(buggy)       # correct fix
patch_b = buggy.replace("-", "*") # wrong fix

winner = heuristic_winner(buggy, patch_a, patch_b, expected_fix=expected)
print(f"=== Heuristic winner: {winner} ===")
print()

# ── 4. Persist a run to a temporary DB ────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

init_db(db_path)
run_id = save_run(
    model_a="mock-model-a",
    model_b="mock-model-b",
    buggy_code=buggy,
    patch_a=patch_a,
    patch_b=patch_b,
    winner=winner,
    scenario_id=sc["id"],
    db_path=db_path,
)
print(f"=== Saved run #{run_id} to {db_path} ===")
print(f"    Total runs in DB: {run_count(db_path)}")
runs = get_runs(limit=1, db_path=db_path)
print(f"    Latest winner: {runs[0]['winner']}")

os.unlink(db_path)
