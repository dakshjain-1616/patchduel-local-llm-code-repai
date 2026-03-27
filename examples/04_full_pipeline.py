"""
04_full_pipeline.py — End-to-end PatchDuel pipeline.

Runs the complete workflow from start to finish:
  1. Load all built-in bug scenarios
  2. Generate patches (mock mode — no Ollama needed)
  3. Score and rank each patch
  4. Determine heuristic winners
  5. Persist all results to SQLite
  6. Print aggregate statistics
  7. Export results to JSON

Run with Ollama available and set USE_LIVE=true to test against real models.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
import time

from patchduel_local_llm_.scenarios import get_all_scenarios, mock_patch
from patchduel_local_llm_.diff_engine import render_html_diff, diff_summary
from patchduel_local_llm_.stats import (
    compute_quality_score,
    heuristic_winner,
    aggregate_session_stats,
    format_stats_line,
)
from patchduel_local_llm_.database import init_db, save_run, export_runs_json, run_count
from patchduel_local_llm_.ollama_client import is_ollama_running, list_models, repair_code_timed

# ── Configuration ─────────────────────────────────────────────────────────────
USE_LIVE = os.environ.get("USE_LIVE", "false").lower() == "true"

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    DB_PATH = f.name
os.environ["DB_PATH"] = DB_PATH

# ── Determine models ──────────────────────────────────────────────────────────
if USE_LIVE and is_ollama_running():
    available = list_models()
    model_a = available[0] if available else "llama3.2"
    model_b = available[1] if len(available) > 1 else "mistral-nemo"
    print(f"[LIVE MODE] Using {model_a} vs {model_b}")
else:
    model_a = "mock-model-a"
    model_b = "mock-model-b"
    print("[MOCK MODE] Using deterministic mock responses")

# ── Pipeline ──────────────────────────────────────────────────────────────────
init_db(DB_PATH)
scenarios = get_all_scenarios()
results = []

print(f"\nRunning {len(scenarios)} scenarios...\n")

for s in scenarios:
    buggy = s["buggy_code"]
    expected = s.get("expected_fix")

    # Get patches
    t0 = time.time()
    if USE_LIVE and is_ollama_running():
        patch_a, err_a, dur_a, tok_a = repair_code_timed(model_a, buggy)
        patch_b, err_b, dur_b, tok_b = repair_code_timed(model_b, buggy)
    else:
        patch_a = mock_patch(buggy)
        patch_b = buggy  # mock model B returns code unchanged
        err_a = err_b = None
        dur_a = dur_b = time.time() - t0
        tok_a = tok_b = 0

    # Score patches
    qa = compute_quality_score(buggy, patch_a, expected_fix=expected)
    qb = compute_quality_score(buggy, patch_b, expected_fix=expected)

    # Determine winner
    winner = heuristic_winner(buggy, patch_a, patch_b, expected_fix=expected)

    # Save to DB
    run_id = save_run(
        model_a=model_a,
        model_b=model_b,
        buggy_code=buggy,
        patch_a=patch_a,
        patch_b=patch_b,
        diff_html_a=render_html_diff(buggy, patch_a),
        diff_html_b=render_html_diff(buggy, patch_b),
        winner=winner,
        duration_a=dur_a,
        duration_b=dur_b,
        tokens_a=tok_a,
        tokens_b=tok_b,
        provider_a="mock" if not USE_LIVE else "ollama",
        provider_b="mock" if not USE_LIVE else "ollama",
        scenario_id=s["id"],
        db_path=DB_PATH,
    )

    print(f"  [{s['id']}] {s['name']}")
    print(f"    A quality={qa}/100  B quality={qb}/100  winner={winner}")

    results.append({
        "id": s["id"],
        "winner": winner,
        "duration_a": dur_a,
        "duration_b": dur_b,
        "tokens_a": tok_a,
        "tokens_b": tok_b,
    })

# ── Aggregate statistics ──────────────────────────────────────────────────────
agg = aggregate_session_stats(results)
print(f"\n{'='*50}")
print(f"  Total scenarios : {agg['total']}")
print(f"  {model_a} wins  : {agg['wins_a']}")
print(f"  {model_b} wins  : {agg['wins_b']}")
print(f"  Ties            : {agg['ties']}")
print(f"  Both wrong      : {agg['both_wrong']}")
print(f"  Avg time A      : {agg['avg_duration_a']:.3f}s")
print(f"  Avg time B      : {agg['avg_duration_b']:.3f}s")
print(f"  DB runs saved   : {run_count(DB_PATH)}")

# ── Export JSON ───────────────────────────────────────────────────────────────
json_export = export_runs_json(DB_PATH)
parsed = json.loads(json_export)
print(f"\nExported {len(parsed)} run(s) to JSON ({len(json_export)} bytes)")

os.unlink(DB_PATH)
print("\nDone. (temp DB cleaned up)")
