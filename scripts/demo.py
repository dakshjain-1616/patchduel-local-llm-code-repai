"""
demo.py — PatchDuel standalone demo runner.

Works fully offline/mock when Ollama is not running (or DEMO_MODE=true).
Always writes output files to outputs/:
  - outputs/results.json     — structured benchmark results (with timing + tokens)
  - outputs/report.html      — visual HTML diff report
  - outputs/summary.txt      — plain-text summary

Usage:
    python demo.py
    DEMO_MODE=true python demo.py        # force mock mode
    DEFAULT_MODEL_A=llama3.2 python demo.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Env ────────────────────────────────────────────────────────────────────────
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEFAULT_MODEL_A = os.getenv("DEFAULT_MODEL_A", "llama3.2")
DEFAULT_MODEL_B = os.getenv("DEFAULT_MODEL_B", "mistral-nemo")
DB_PATH = os.getenv("DB_PATH", str(ROOT / "patchduel_demo.db"))

# Override DB for demo so it doesn't pollute the main DB
os.environ["DB_PATH"] = DB_PATH

from patchduel_local_llm_.database import init_db, save_run
from patchduel_local_llm_.diff_engine import render_html_diff, diff_stats, diff_summary
from patchduel_local_llm_.ollama_client import is_ollama_running, repair_code_timed
from patchduel_local_llm_.scenarios import SCENARIOS, mock_patch
from patchduel_local_llm_.stats import aggregate_session_stats, format_stats_line, heuristic_winner

# ── Test cases (drawn from the scenario library) ───────────────────────────────
# Use the first 5 scenarios for the demo run
DEMO_SCENARIO_IDS = ["sc-001", "sc-002", "sc-003", "sc-005", "sc-006"]
TEST_CASES = [s for s in SCENARIOS if s["id"] in DEMO_SCENARIO_IDS]


def get_patch(model: str, buggy_code: str, use_mock: bool) -> tuple[str, str | None, float, int]:
    """Get patch from model or mock. Returns (patch, error, duration_secs, tokens)."""
    if use_mock:
        patch = mock_patch(buggy_code)
        return patch, None, 0.001, 0
    return repair_code_timed(model, buggy_code)


def run_demo() -> None:
    """Run benchmark scenarios in mock or live mode and write outputs/ files."""
    print("=" * 60)
    print("  PatchDuel — Local LLM Code Repair Arena  |  Demo Run")
    print("=" * 60)

    ollama_up = is_ollama_running()
    use_mock = DEMO_MODE or not ollama_up

    if use_mock:
        print(
            "\n[DEMO MODE] Ollama not detected (or DEMO_MODE=true). "
            "Using deterministic mock responses.\n"
        )
        model_a = DEFAULT_MODEL_A
        model_b = DEFAULT_MODEL_B
    else:
        from patchduel_local_llm_.ollama_client import list_models
        available = list_models()
        model_a = available[0] if available else DEFAULT_MODEL_A
        model_b = available[1] if len(available) > 1 else DEFAULT_MODEL_B
        print(f"\n[LIVE MODE] Ollama running — using {model_a} vs {model_b}\n")

    init_db(DB_PATH)

    results = []
    run_ids = []
    timestamp_run = datetime.now(timezone.utc).isoformat()

    for tc in TEST_CASES:
        print(f"─── {tc['id']}: {tc['description']}")

        patch_a, err_a, dur_a, tok_a = get_patch(model_a, tc["buggy_code"], use_mock)
        patch_b, err_b, dur_b, tok_b = get_patch(model_b, tc["buggy_code"], use_mock)

        if err_a:
            print(f"    ✗ {model_a} error: {err_a}")
            patch_a = f"# ERROR: {err_a}"
        if err_b:
            print(f"    ✗ {model_b} error: {err_b}")
            patch_b = f"# ERROR: {err_b}"

        diff_html_a = render_html_diff(tc["buggy_code"], patch_a)
        diff_html_b = render_html_diff(tc["buggy_code"], patch_b)

        stats_a = diff_stats(tc["buggy_code"], patch_a)
        stats_b = diff_stats(tc["buggy_code"], patch_b)

        summary_a = diff_summary(tc["buggy_code"], patch_a)
        summary_b = diff_summary(tc["buggy_code"], patch_b)

        print(f"    {model_a}: {summary_a}  [{dur_a:.2f}s | {tok_a} tok]")
        print(f"    {model_b}: {summary_b}  [{dur_b:.2f}s | {tok_b} tok]")

        winner = heuristic_winner(
            tc["buggy_code"], patch_a, patch_b,
            expected_fix=tc.get("expected_fix"),
        )
        print(f"    → Heuristic winner: {winner}")

        run_id = save_run(
            model_a=model_a,
            model_b=model_b,
            buggy_code=tc["buggy_code"],
            patch_a=patch_a,
            patch_b=patch_b,
            diff_html_a=diff_html_a,
            diff_html_b=diff_html_b,
            winner=winner,
            notes=f"Demo run — {tc['description']}",
            duration_a=dur_a,
            duration_b=dur_b,
            tokens_a=tok_a,
            tokens_b=tok_b,
            provider_a="mock" if use_mock else "ollama",
            provider_b="mock" if use_mock else "ollama",
            scenario_id=tc["id"],
            db_path=DB_PATH,
        )
        run_ids.append(run_id)

        results.append({
            "id": tc["id"],
            "description": tc["description"],
            "language": tc.get("language", "python"),
            "run_id": run_id,
            "model_a": model_a,
            "model_b": model_b,
            "buggy_code": tc["buggy_code"],
            "patch_a": patch_a,
            "patch_b": patch_b,
            "diff_stats_a": stats_a,
            "diff_stats_b": stats_b,
            "summary_a": summary_a,
            "summary_b": summary_b,
            "winner": winner,
            "duration_a": round(dur_a, 3),
            "duration_b": round(dur_b, 3),
            "tokens_a": tok_a,
            "tokens_b": tok_b,
        })
        print()

    # ── Aggregate stats ─────────────────────────────────────────────────────────
    agg = aggregate_session_stats(results)

    # ── Write outputs/results.json ──────────────────────────────────────────────
    output_data = {
        "meta": {
            "generated_at": timestamp_run,
            "mode": "mock" if use_mock else "live",
            "model_a": model_a,
            "model_b": model_b,
            "total_cases": len(TEST_CASES),
            "db_path": DB_PATH,
            "aggregate": agg,
        },
        "results": results,
    }
    results_path = OUTPUTS_DIR / "results.json"
    results_path.write_text(json.dumps(output_data, indent=2))
    print(f"  Saved: {results_path}")

    # ── Write outputs/report.html ───────────────────────────────────────────────
    html_report = _build_html_report(model_a, model_b, results, timestamp_run, use_mock, agg)
    report_path = OUTPUTS_DIR / "report.html"
    report_path.write_text(html_report)
    print(f"  Saved: {report_path}")

    # ── Write outputs/summary.txt ───────────────────────────────────────────────
    summary_lines = [
        "PatchDuel Demo Run Summary",
        "=" * 40,
        f"Generated : {timestamp_run}",
        f"Mode      : {'mock' if use_mock else 'live'}",
        f"Model A   : {model_a}  ({agg['wins_a']} wins)",
        f"Model B   : {model_b}  ({agg['wins_b']} wins)",
        f"Ties      : {agg['ties']}",
        f"Both Wrong: {agg['both_wrong']}",
        f"Total cases: {len(TEST_CASES)}",
        "",
        "Per-case results:",
    ]
    for r in results:
        timing = format_stats_line(
            r["duration_a"], r["duration_b"],
            r["tokens_a"], r["tokens_b"],
            model_a, model_b,
        )
        summary_lines.append(
            f"  [{r['id']}] {r['description'][:45]}"
            f" → Winner: {r['winner']}"
        )
        summary_lines.append(f"    {timing}")
    summary_lines += [
        "",
        f"SQLite DB : {DB_PATH}",
        f"Run IDs   : {run_ids}",
    ]
    summary_text = "\n".join(summary_lines)
    summary_path = OUTPUTS_DIR / "summary.txt"
    summary_path.write_text(summary_text)
    print(f"  Saved: {summary_path}")

    print()
    print(summary_text)
    print()
    print("Demo complete. Open outputs/report.html to view the visual diff report.")


def _build_html_report(
    model_a: str,
    model_b: str,
    results: list[dict],
    timestamp: str,
    mock: bool,
    agg: dict,
) -> str:
    """Build a self-contained HTML diff report with timing and aggregate stats."""
    import html as html_mod

    mode_badge = (
        '<span style="background:#1e3a2a;color:#3fb950;padding:2px 8px;'
        'border-radius:12px;font-size:0.8em;">MOCK MODE</span>'
        if mock
        else '<span style="background:#1e2a3a;color:#58a6ff;padding:2px 8px;'
        'border-radius:12px;font-size:0.8em;">LIVE MODE</span>'
    )

    # Aggregate stats bar
    agg_html = (
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
        f"padding:12px;margin-bottom:24px;font-size:0.85em;color:#8b949e;'>"
        f"<strong style='color:#e0e0e0;'>Aggregate Results</strong> &nbsp;|&nbsp; "
        f"Model A wins: <strong style='color:#3fb950;'>{agg['wins_a']}</strong> &nbsp;|&nbsp; "
        f"Model B wins: <strong style='color:#58a6ff;'>{agg['wins_b']}</strong> &nbsp;|&nbsp; "
        f"Ties: {agg['ties']} &nbsp;|&nbsp; "
        f"Both Wrong: {agg['both_wrong']} &nbsp;|&nbsp; "
        f"Avg time A: {agg['avg_duration_a']:.2f}s &nbsp;|&nbsp; "
        f"Avg time B: {agg['avg_duration_b']:.2f}s"
        f"</div>"
    )

    case_html = []
    for r in results:
        _winner_badge = lambda role: (
            '<span style="background:#1e3a2a;color:#3fb950;padding:1px 6px;'
            'border-radius:8px;font-size:0.75em;margin-left:6px;">WINNER</span>'
            if r["winner"] == f"Model {role}" else ""
        )
        buggy_escaped = html_mod.escape(r["buggy_code"])
        timing_a = f"{r['duration_a']:.2f}s | {r['tokens_a']} tok"
        timing_b = f"{r['duration_b']:.2f}s | {r['tokens_b']} tok"

        case_html.append(f"""
<div class="case-card">
  <h2>{r['id']}: {html_mod.escape(r['description'])}</h2>
  <div class="meta-row">
    <span>Run ID: #{r['run_id']}</span>
    <span>Language: {r.get('language', 'python')}</span>
    <span>Winner: <strong>{r['winner']}</strong></span>
  </div>

  <h3>Buggy Input</h3>
  <pre class="code-block">{buggy_escaped}</pre>

  <div class="side-by-side">
    <div class="panel">
      <h3>{html_mod.escape(r['model_a'])}{_winner_badge('A')}</h3>
      <div class="stats">{html_mod.escape(r['summary_a'])} &nbsp;|&nbsp; ⏱ {timing_a}</div>
      {render_html_diff(r['buggy_code'], r['patch_a'])}
    </div>
    <div class="panel">
      <h3>{html_mod.escape(r['model_b'])}{_winner_badge('B')}</h3>
      <div class="stats">{html_mod.escape(r['summary_b'])} &nbsp;|&nbsp; ⏱ {timing_b}</div>
      {render_html_diff(r['buggy_code'], r['patch_b'])}
    </div>
  </div>
</div>
""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PatchDuel Report — {timestamp[:10]}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 12px; }}
  h2 {{ color: #e0e0e0; margin-top: 0; }}
  h3 {{ color: #8b949e; font-size: 0.95em; margin: 12px 0 6px; }}
  .header-meta {{ color: #8b949e; font-size: 0.85em; margin-bottom: 24px; }}
  .case-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
                padding: 20px; margin-bottom: 28px; }}
  .meta-row {{ display: flex; gap: 20px; font-size: 0.82em; color: #8b949e;
               margin-bottom: 16px; flex-wrap: wrap; }}
  .code-block {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                  padding: 12px; font-size: 13px; overflow-x: auto;
                  white-space: pre; color: #e0e0e0; }}
  .side-by-side {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
                    margin-top: 16px; }}
  .panel {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
             padding: 12px; }}
  .stats {{ font-size: 0.82em; color: #8b949e; margin-bottom: 8px;
             font-family: monospace; }}
  @media (max-width: 768px) {{ .side-by-side {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>🥊 PatchDuel — Code Repair Report</h1>
<div class="header-meta">
  Generated: {timestamp} &nbsp;|&nbsp;
  {mode_badge} &nbsp;|&nbsp;
  Model A: <strong>{html_mod.escape(model_a)}</strong> &nbsp;|&nbsp;
  Model B: <strong>{html_mod.escape(model_b)}</strong> &nbsp;|&nbsp;
  Cases: {len(results)}
</div>

{agg_html}

{"".join(case_html)}

<footer style="color:#30363d;font-size:0.8em;text-align:center;margin-top:40px;">
  Generated by PatchDuel · Local LLM Code Repair Arena
</footer>
</body>
</html>"""


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
