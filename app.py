"""PatchDuel — Local LLM Code Repair Arena.

Gradio 4.x app that pits two LLMs head-to-head on code repair tasks,
renders git-style diffs, saves reproducible evaluation logs to SQLite,
and exports results. Supports Ollama (local), OpenRouter (cloud), and
Mock (demo/offline) providers.
"""
import os
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from patchduel_local_llm_.database import (
    delete_run,
    export_runs_csv,
    export_runs_json,
    get_leaderboard,
    get_runs,
    init_db,
    run_count,
    save_run,
)
from patchduel_local_llm_.diff_engine import diff_summary, render_html_diff
from patchduel_local_llm_.ollama_client import (
    is_ollama_running,
    list_models as ollama_list_models,
    repair_code_timed as ollama_repair_timed,
)
from patchduel_local_llm_.openrouter_client import (
    is_configured as openrouter_configured,
    list_models as openrouter_list_models,
    repair_code_timed as openrouter_repair_timed,
)
from patchduel_local_llm_.scenarios import (
    get_all_scenarios,
    get_scenario_by_id,
    mock_patch,
    scenario_choices,
)
from patchduel_local_llm_.stats import compute_quality_score, format_stats_line

# ── Environment ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_A = os.getenv("DEFAULT_MODEL_A", "llama3.2")
DEFAULT_MODEL_B = os.getenv("DEFAULT_MODEL_B", "mistral-nemo")
DEFAULT_OR_MODEL_A = os.getenv("DEFAULT_OPENROUTER_MODEL_A", "openai/gpt-5.4-mini")
DEFAULT_OR_MODEL_B = os.getenv("DEFAULT_OPENROUTER_MODEL_B", "mistralai/mistral-small-2603")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# ── Bootstrap ──────────────────────────────────────────────────────────────────
init_db()

# ── Constants ─────────────────────────────────────────────────────────────────
_PROVIDERS = ["ollama", "openrouter", "mock"]

_PLACEHOLDER_DIFF = """
<div style="padding:20px;color:#8b949e;font-family:monospace;
            background:#0d1117;border-radius:6px;border:1px solid #30363d;">
  Diff will appear here after generation.
</div>
"""

_LOADING_DIFF = """
<div style="padding:20px;color:#8b949e;font-family:monospace;
            background:#0d1117;border-radius:6px;border:1px solid #30363d;">
  ⏳ Generating patch…
</div>
"""

_ERROR_STYLE = (
    "padding:12px;background:#2a0d0d;color:#f85149;"
    "border-radius:6px;font-family:monospace;border:1px solid #f85149;"
)

# ── Provider / model helpers ───────────────────────────────────────────────────

def _models_for_provider(provider: str) -> list[str]:
    """Return available model names for the given provider."""
    if provider == "ollama":
        models = ollama_list_models()
        return models if models else [DEFAULT_MODEL_A, DEFAULT_MODEL_B]
    if provider == "openrouter":
        return openrouter_list_models()
    # mock
    return ["mock-demo"]


def _provider_status() -> str:
    """Return a one-line status string showing which providers are online."""
    parts = []
    if is_ollama_running():
        n = len(ollama_list_models())
        parts.append(f"Ollama ✓ ({n} models)")
    else:
        parts.append("Ollama ✗ (offline)")
    if openrouter_configured():
        parts.append("OpenRouter ✓ (key set)")
    else:
        parts.append("OpenRouter ✗ (no key)")
    if DEMO_MODE:
        parts.append("DEMO_MODE=true")
    return "  |  ".join(parts)


# ── Unified patch getter ───────────────────────────────────────────────────────

def _get_patch(
    provider: str, model: str, buggy_code: str
) -> tuple[str, str | None, float, int]:
    """Returns (patch, error, duration_secs, tokens) for any provider."""
    if provider == "mock" or DEMO_MODE:
        t0 = time.time()
        patch = mock_patch(buggy_code)
        return patch, None, time.time() - t0, 0
    if provider == "openrouter":
        return openrouter_repair_timed(model, buggy_code)
    # default: ollama
    return ollama_repair_timed(model, buggy_code)


# ── Duel tab callbacks ─────────────────────────────────────────────────────────

def update_provider(provider: str) -> gr.Dropdown:
    """Update model dropdown when provider changes."""
    models = _models_for_provider(provider)
    return gr.Dropdown(choices=models, value=models[0] if models else "")


def load_scenario(scenario_id: str) -> tuple[str, str]:
    """Load buggy code and a card HTML snippet for the selected scenario."""
    if scenario_id == "custom" or not scenario_id:
        return (
            "",
            "<div style='padding:8px;color:#8b949e;font-size:0.85em;'>"
            "Paste any buggy code into the input below.</div>",
        )
    s = get_scenario_by_id(scenario_id)
    if not s:
        return "", ""
    tags_html = "".join(
        f"<span style='background:#1e2a3a;color:#58a6ff;padding:2px 8px;"
        f"border-radius:10px;font-size:0.75em;margin-right:4px;'>{t}</span>"
        for t in s.get("tags", [])
    )
    card = (
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
        f"padding:12px;font-size:0.85em;'>"
        f"<strong style='color:#e0e0e0;'>{s['name']}</strong> "
        f"<span style='color:#8b949e;'>— {s['description']}</span><br/>"
        f"<div style='margin-top:6px;'>{tags_html}</div>"
        f"</div>"
    )
    return s["buggy_code"], card


def generate_patches(
    provider_a: str, model_a: str,
    provider_b: str, model_b: str,
    scenario_id: str, buggy_code: str,
):
    """
    Generator — yields progressive UI updates as each model completes.
    Outputs: patch_a, diff_a, qa_a, patch_b, diff_b, qb_b, stats_out,
             state_diff_a, state_diff_b, state_dur_a, state_dur_b,
             state_tok_a, state_tok_b
    """
    if not buggy_code.strip():
        empty = "<div style='padding:12px;color:#8b949e;'>No code provided.</div>"
        yield ("", empty, "", "", empty, "", "", "", "", 0.0, 0.0, 0, 0)
        return

    # ── Phase 1: show "generating" state ──────────────────────────────────────
    yield (
        "⏳ Generating…", _LOADING_DIFF, "",
        "⏳ Waiting…", _PLACEHOLDER_DIFF, "",
        "Calling models…",
        "", "", 0.0, 0.0, 0, 0,
    )

    # ── Phase 2: Model A ──────────────────────────────────────────────────────
    patch_a, err_a, dur_a, tok_a = _get_patch(provider_a, model_a, buggy_code)
    if err_a:
        html_a = f"<div style='{_ERROR_STYLE}'>Error calling {model_a}:<br>{err_a}</div>"
        patch_a_display = f"# ERROR: {err_a}"
        summary_a = "N/A"
        qa = 0
    else:
        html_a = render_html_diff(buggy_code, patch_a)
        patch_a_display = patch_a
        summary_a = diff_summary(buggy_code, patch_a)
        qa = compute_quality_score(buggy_code, patch_a)

    yield (
        patch_a_display, html_a, f"Quality: {qa}/100  |  {summary_a}",
        "⏳ Generating…", _LOADING_DIFF, "",
        f"⏱ {model_a.split('/')[-1]}: {dur_a:.1f}s | {tok_a or '?'} tok",
        html_a, "", dur_a, 0.0, tok_a, 0,
    )

    # ── Phase 3: Model B ──────────────────────────────────────────────────────
    patch_b, err_b, dur_b, tok_b = _get_patch(provider_b, model_b, buggy_code)
    if err_b:
        html_b = f"<div style='{_ERROR_STYLE}'>Error calling {model_b}:<br>{err_b}</div>"
        patch_b_display = f"# ERROR: {err_b}"
        summary_b = "N/A"
        qb = 0
    else:
        html_b = render_html_diff(buggy_code, patch_b)
        patch_b_display = patch_b
        summary_b = diff_summary(buggy_code, patch_b)
        qb = compute_quality_score(buggy_code, patch_b)

    stats_line = format_stats_line(dur_a, dur_b, tok_a, tok_b, model_a, model_b)

    yield (
        patch_a_display, html_a, f"Quality: {qa}/100  |  {summary_a}",
        patch_b_display, html_b, f"Quality: {qb}/100  |  {summary_b}",
        stats_line,
        html_a, html_b, dur_a, dur_b, tok_a, tok_b,
    )


def save_current_run(
    provider_a, model_a, provider_b, model_b,
    scenario_id, buggy_code,
    patch_a, patch_b,
    diff_html_a, diff_html_b,
    dur_a, dur_b, tok_a, tok_b,
    winner, notes,
):
    """Persist the current duel to SQLite and return a confirmation message."""
    if not buggy_code.strip():
        return "Nothing to save — enter code and generate patches first."
    if not (patch_a or "").strip() and not (patch_b or "").strip():
        return "No patches generated yet."

    run_id = save_run(
        model_a=model_a,
        model_b=model_b,
        buggy_code=buggy_code,
        patch_a=patch_a or "",
        patch_b=patch_b or "",
        diff_html_a=diff_html_a or "",
        diff_html_b=diff_html_b or "",
        winner=winner or None,
        notes=notes or None,
        duration_a=float(dur_a) if dur_a else None,
        duration_b=float(dur_b) if dur_b else None,
        tokens_a=int(tok_a) if tok_a else 0,
        tokens_b=int(tok_b) if tok_b else 0,
        provider_a=provider_a or "ollama",
        provider_b=provider_b or "ollama",
        scenario_id=scenario_id if scenario_id != "custom" else None,
    )
    total = run_count()
    return f"✔ Run #{run_id} saved  ({total} total in DB)"


# ── History tab callbacks ──────────────────────────────────────────────────────

def load_history():
    """Fetch the 100 most recent runs and format them for the history table."""
    runs = get_runs(100)
    if not runs:
        return []
    rows = []
    for r in runs:
        preview = r["buggy_code"].replace("\n", " ")
        preview = preview[:55] + "…" if len(preview) > 55 else preview
        tok_a = r.get("tokens_a") or 0
        tok_b = r.get("tokens_b") or 0
        dur_a = r.get("duration_a")
        dur_b = r.get("duration_b")
        timing = (
            f"A:{dur_a:.1f}s B:{dur_b:.1f}s" if dur_a and dur_b else "—"
        )
        rows.append([
            r["id"],
            r["timestamp"][:19].replace("T", " "),
            r.get("provider_a", "ollama"),
            r["model_a"],
            r.get("provider_b", "ollama"),
            r["model_b"],
            preview,
            r.get("winner") or "—",
            timing,
            f"{tok_a + tok_b}" if (tok_a or tok_b) else "—",
            r.get("notes") or "",
        ])
    return rows


def do_export_csv() -> str:
    """Write CSV to a temp file and return the path for gr.File download."""
    csv_text = export_runs_csv()
    if not csv_text:
        return None  # type: ignore[return-value]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="patchduel_", delete=False
    )
    f.write(csv_text)
    f.close()
    return f.name


def do_export_json() -> str:
    """Write JSON to a temp file and return the path for gr.File download."""
    json_text = export_runs_json()
    if not json_text or json_text == "[]":
        return None  # type: ignore[return-value]
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="patchduel_", delete=False
    )
    f.write(json_text)
    f.close()
    return f.name


def delete_run_by_id(run_id_str: str) -> tuple[str, list]:
    """Delete a run by its numeric ID and return a status message and refreshed history."""
    try:
        run_id = int(str(run_id_str).strip())
    except (ValueError, TypeError):
        return "Invalid run ID.", load_history()
    ok = delete_run(run_id)
    msg = f"✔ Run #{run_id} deleted." if ok else f"Run #{run_id} not found."
    return msg, load_history()


# ── Leaderboard callback ───────────────────────────────────────────────────────

def load_leaderboard():
    """Return per-model win/loss/tie stats formatted for the leaderboard table."""
    board = get_leaderboard()
    if not board:
        return []
    return [
        [e["model"], e["runs"], e["wins"], e["ties"], e["losses"], e["win_rate"]]
        for e in board
    ]


# ── Batch run callback ─────────────────────────────────────────────────────────

def run_batch(
    provider_a: str, model_a: str,
    provider_b: str, model_b: str,
    progress=gr.Progress(),
):
    """Run all built-in scenarios against a model pair, yielding incremental table rows."""
    scenarios = get_all_scenarios()
    rows = []
    progress(0, desc="Starting batch…")
    for i, s in enumerate(scenarios):
        progress(i / len(scenarios), desc=f"[{s['id']}] {s['name']}")
        patch_a, err_a, dur_a, tok_a = _get_patch(provider_a, model_a, s["buggy_code"])
        patch_b, err_b, dur_b, tok_b = _get_patch(provider_b, model_b, s["buggy_code"])
        from patchduel_local_llm_.stats import heuristic_winner
        winner = heuristic_winner(
            s["buggy_code"], patch_a or "", patch_b or "",
            expected_fix=s.get("expected_fix"),
        )
        qa = compute_quality_score(s["buggy_code"], patch_a or "", s.get("expected_fix"))
        qb = compute_quality_score(s["buggy_code"], patch_b or "", s.get("expected_fix"))
        rows.append([
            s["id"],
            s["name"],
            winner,
            f"{qa}/100",
            f"{qb}/100",
            f"{dur_a:.1f}s / {tok_a or '?'} tok",
            f"{dur_b:.1f}s / {tok_b or '?'} tok",
        ])
        yield rows, f"✓ {i + 1}/{len(scenarios)} — last: {s['name']}"
    progress(1.0, desc="Done!")
    yield rows, f"Batch complete — {len(scenarios)} scenarios evaluated."


# ── Session stats helper ───────────────────────────────────────────────────────

def _update_session_stats(
    turns: int, tok_a: int, tok_b: int, dur_a: float, dur_b: float
) -> tuple[int, str]:
    """Increment the duel counter and reformat the live session stats markdown."""
    new_turns = turns + 1
    total_tok = (tok_a or 0) + (tok_b or 0)
    avg_time = ((dur_a or 0.0) + (dur_b or 0.0)) / 2
    label = "duel" if new_turns == 1 else "duels"
    return (
        new_turns,
        f"**Session stats** — {new_turns} {label} · "
        f"{total_tok} tokens · {avg_time:.1f}s avg response time",
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

_CSS = """
#header { text-align:center; padding:8px 0 4px; }
.panel-label { font-size:1em; font-weight:600; color:#e0e0e0; margin-bottom:4px; }
.stats-row { font-family:monospace; font-size:0.9em; color:#8b949e; }
.quality-label { font-size:0.82em; color:#8b949e; font-family:monospace; }
"""

with gr.Blocks(title="PatchDuel: LLM Code Repair Arena", theme=gr.themes.Soft(), css=_CSS) as demo:
    demo.queue()  # Required for streaming generators

    gr.Markdown(
        "# 🥊 PatchDuel: Local LLM Code Repair Arena\n"
        "Compare two LLMs head-to-head on bug fixing — local or cloud. "
        "Inspect git-style diffs and save reproducible evaluation logs.\n\n"
        "*Built autonomously using [NEO](https://heyneo.so) — your autonomous AI Agent*",
        elem_id="header",
    )

    # ── Provider status bar ───────────────────────────────────────────────────
    provider_status = gr.Textbox(
        value=_provider_status(),
        label="Provider Status",
        interactive=False,
        max_lines=1,
    )
    refresh_status_btn = gr.Button("↺ Refresh Status", variant="secondary", size="sm")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — DUEL
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("⚔️ Duel"):

            # ── Model selectors ───────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Model A**", elem_classes=["panel-label"])
                    provider_a_dd = gr.Dropdown(
                        choices=_PROVIDERS,
                        value="ollama" if not DEMO_MODE else "mock",
                        label="Provider A",
                        interactive=True,
                    )
                    _init_a = _models_for_provider("mock" if DEMO_MODE else "ollama")
                    model_a_dd = gr.Dropdown(
                        choices=_init_a,
                        value=_init_a[0] if _init_a else DEFAULT_MODEL_A,
                        label="Model A",
                        interactive=True,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**Model B**", elem_classes=["panel-label"])
                    provider_b_dd = gr.Dropdown(
                        choices=_PROVIDERS,
                        value="ollama" if not DEMO_MODE else "mock",
                        label="Provider B",
                        interactive=True,
                    )
                    _init_b = _init_a
                    model_b_dd = gr.Dropdown(
                        choices=_init_b,
                        value=_init_b[1] if len(_init_b) > 1 else DEFAULT_MODEL_B,
                        label="Model B",
                        interactive=True,
                    )

            # ── Scenario picker ───────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=2):
                    scenario_dd = gr.Dropdown(
                        choices=scenario_choices(),
                        value="custom",
                        label="Scenario",
                        interactive=True,
                    )
                with gr.Column(scale=3):
                    scenario_card = gr.HTML(
                        value="<div style='padding:8px;color:#8b949e;font-size:0.85em;'>"
                              "Select a scenario above or paste custom code below.</div>"
                    )

            # ── Code input ────────────────────────────────────────────────────
            buggy_input = gr.Textbox(
                label="Buggy Code",
                placeholder="Paste your buggy code here, or select a scenario above…",
                lines=10,
                value="def add(a, b):\n    return a - b  # Bug: should be a + b",
            )

            generate_btn = gr.Button("⚡ Generate Patches", variant="primary", size="lg")

            # ── Real-time stats ───────────────────────────────────────────────
            stats_out = gr.Textbox(
                label="⏱ Timing & Tokens",
                interactive=False,
                max_lines=1,
                elem_classes=["stats-row"],
            )
            live_stats_md = gr.Markdown(
                "**Session stats** — 0 duels · 0 tokens · 0.0s avg response time",
                elem_classes=["stats-row"],
            )

            # ── Diff output ───────────────────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model A — Patch & Diff")
                    patch_a_out = gr.Textbox(
                        label="Fixed Code (A)", lines=8, interactive=False
                    )
                    quality_a_out = gr.Textbox(
                        label="Quality & Diff Stats (A)", interactive=False, max_lines=1,
                        elem_classes=["quality-label"],
                    )
                    diff_a_out = gr.HTML(value=_PLACEHOLDER_DIFF, label="Diff View (A)")

                with gr.Column():
                    gr.Markdown("### Model B — Patch & Diff")
                    patch_b_out = gr.Textbox(
                        label="Fixed Code (B)", lines=8, interactive=False
                    )
                    quality_b_out = gr.Textbox(
                        label="Quality & Diff Stats (B)", interactive=False, max_lines=1,
                        elem_classes=["quality-label"],
                    )
                    diff_b_out = gr.HTML(value=_PLACEHOLDER_DIFF, label="Diff View (B)")

            # ── Hidden state ──────────────────────────────────────────────────
            state_diff_a = gr.State("")
            state_diff_b = gr.State("")
            state_dur_a = gr.State(0.0)
            state_dur_b = gr.State(0.0)
            state_tok_a = gr.State(0)
            state_tok_b = gr.State(0)
            state_scenario = gr.State("custom")
            state_turns = gr.State(0)

            # ── Evaluation ────────────────────────────────────────────────────
            with gr.Row():
                winner_radio = gr.Radio(
                    choices=["Model A", "Model B", "Tie", "Both Wrong"],
                    label="Winner",
                    value=None,
                    scale=2,
                )
                notes_box = gr.Textbox(
                    label="Notes",
                    placeholder="Optional evaluation notes…",
                    lines=2,
                    scale=3,
                )

            with gr.Row():
                save_btn = gr.Button("💾 Save Run", variant="secondary")
                save_status = gr.Textbox(
                    label="Save Status", interactive=False, max_lines=1, scale=3
                )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — HISTORY
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📋 History"):
            with gr.Row():
                refresh_hist_btn = gr.Button("↺ Refresh", variant="secondary", size="sm")
                export_csv_btn = gr.Button("⬇ Export CSV", variant="secondary", size="sm")
                export_json_btn = gr.Button("⬇ Export JSON", variant="secondary", size="sm")

            csv_file_out = gr.File(label="CSV Download", visible=False)
            json_file_out = gr.File(label="JSON Download", visible=False)

            history_table = gr.DataFrame(
                headers=[
                    "ID", "Timestamp (UTC)",
                    "Prov A", "Model A", "Prov B", "Model B",
                    "Code Preview", "Winner", "Timing", "Tokens", "Notes",
                ],
                label="Saved Runs (newest first)",
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                delete_id_box = gr.Textbox(
                    label="Delete Run by ID",
                    placeholder="Enter run ID…",
                    max_lines=1,
                    scale=1,
                )
                delete_btn = gr.Button("🗑 Delete", variant="stop", size="sm", scale=0)
                delete_status = gr.Textbox(
                    label="Delete Status", interactive=False, max_lines=1, scale=3
                )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — LEADERBOARD
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown(
                "Win/loss/tie stats across all saved runs. "
                "Only runs where a winner was selected are counted."
            )
            refresh_lb_btn = gr.Button("↺ Refresh", variant="secondary", size="sm")
            leaderboard_table = gr.DataFrame(
                headers=["Model", "Runs", "Wins", "Ties", "Losses", "Win Rate"],
                label="Model Leaderboard",
                interactive=False,
                wrap=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — BATCH RUN
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🧪 Batch Run"):
            gr.Markdown(
                "Run all built-in scenarios against a model pair and see results "
                "at a glance. Uses the same provider/model settings as the Duel tab."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    batch_provider_a = gr.Dropdown(
                        choices=_PROVIDERS,
                        value="mock",
                        label="Provider A",
                    )
                    batch_model_a = gr.Dropdown(
                        choices=["mock-demo"],
                        value="mock-demo",
                        label="Model A",
                    )
                with gr.Column(scale=1):
                    batch_provider_b = gr.Dropdown(
                        choices=_PROVIDERS,
                        value="mock",
                        label="Provider B",
                    )
                    batch_model_b = gr.Dropdown(
                        choices=["mock-demo"],
                        value="mock-demo",
                        label="Model B",
                    )

            batch_run_btn = gr.Button("▶ Run All Scenarios", variant="primary")
            batch_status = gr.Textbox(
                label="Batch Status", interactive=False, max_lines=1
            )
            batch_table = gr.DataFrame(
                headers=[
                    "Scenario ID", "Name", "Winner",
                    "Quality A", "Quality B",
                    "A: Time / Tokens", "B: Time / Tokens",
                ],
                label="Batch Results",
                interactive=False,
                wrap=True,
            )

    # ── Wiring ─────────────────────────────────────────────────────────────────

    refresh_status_btn.click(fn=_provider_status, outputs=[provider_status])

    # Provider → model choices
    provider_a_dd.change(
        fn=lambda p: gr.Dropdown(choices=_models_for_provider(p),
                                 value=_models_for_provider(p)[0]),
        inputs=[provider_a_dd],
        outputs=[model_a_dd],
    )
    provider_b_dd.change(
        fn=lambda p: gr.Dropdown(choices=_models_for_provider(p),
                                 value=_models_for_provider(p)[0]),
        inputs=[provider_b_dd],
        outputs=[model_b_dd],
    )

    # Batch provider → model choices
    batch_provider_a.change(
        fn=lambda p: gr.Dropdown(choices=_models_for_provider(p),
                                 value=_models_for_provider(p)[0]),
        inputs=[batch_provider_a],
        outputs=[batch_model_a],
    )
    batch_provider_b.change(
        fn=lambda p: gr.Dropdown(choices=_models_for_provider(p),
                                 value=_models_for_provider(p)[0]),
        inputs=[batch_provider_b],
        outputs=[batch_model_b],
    )

    # Scenario picker
    scenario_dd.change(
        fn=load_scenario,
        inputs=[scenario_dd],
        outputs=[buggy_input, scenario_card],
    )
    scenario_dd.change(fn=lambda x: x, inputs=[scenario_dd], outputs=[state_scenario])

    # Generate (streaming)
    generate_btn.click(
        fn=generate_patches,
        inputs=[
            provider_a_dd, model_a_dd,
            provider_b_dd, model_b_dd,
            state_scenario, buggy_input,
        ],
        outputs=[
            patch_a_out, diff_a_out, quality_a_out,
            patch_b_out, diff_b_out, quality_b_out,
            stats_out,
            state_diff_a, state_diff_b,
            state_dur_a, state_dur_b,
            state_tok_a, state_tok_b,
        ],
    ).then(
        fn=_update_session_stats,
        inputs=[state_turns, state_tok_a, state_tok_b, state_dur_a, state_dur_b],
        outputs=[state_turns, live_stats_md],
    )

    # Save
    save_btn.click(
        fn=save_current_run,
        inputs=[
            provider_a_dd, model_a_dd,
            provider_b_dd, model_b_dd,
            state_scenario, buggy_input,
            patch_a_out, patch_b_out,
            state_diff_a, state_diff_b,
            state_dur_a, state_dur_b,
            state_tok_a, state_tok_b,
            winner_radio, notes_box,
        ],
        outputs=[save_status],
    )

    # History tab
    refresh_hist_btn.click(fn=load_history, outputs=[history_table])
    export_csv_btn.click(
        fn=do_export_csv,
        outputs=[csv_file_out],
    ).then(fn=lambda: gr.File(visible=True), outputs=[csv_file_out])
    export_json_btn.click(
        fn=do_export_json,
        outputs=[json_file_out],
    ).then(fn=lambda: gr.File(visible=True), outputs=[json_file_out])
    delete_btn.click(
        fn=delete_run_by_id,
        inputs=[delete_id_box],
        outputs=[delete_status, history_table],
    )
    demo.load(fn=load_history, outputs=[history_table])

    # Leaderboard tab
    refresh_lb_btn.click(fn=load_leaderboard, outputs=[leaderboard_table])
    demo.load(fn=load_leaderboard, outputs=[leaderboard_table])

    # Batch run (streaming generator)
    batch_run_btn.click(
        fn=run_batch,
        inputs=[batch_provider_a, batch_model_a, batch_provider_b, batch_model_b],
        outputs=[batch_table, batch_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        share=GRADIO_SHARE,
    )
