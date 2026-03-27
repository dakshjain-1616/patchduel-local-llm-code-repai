# PatchDuel — Local LLM Code Repair Arena
# Public API re-exports for the patchduel_local_llm_ package.

from .database import (
    delete_run,
    export_runs_csv,
    export_runs_json,
    get_leaderboard,
    get_run_by_id,
    get_runs,
    init_db,
    run_count,
    save_run,
)
from .diff_engine import (
    compute_char_diff,
    compute_line_diff,
    diff_stats,
    diff_summary,
    render_html_diff,
)
from .ollama_client import (
    _strip_markdown_fences,
    generate,
    is_ollama_running,
    list_models,
    model_exists,
    repair_code,
    repair_code_timed,
)
from .openrouter_client import (
    is_configured,
    list_models as openrouter_list_models,
    repair_code as openrouter_repair_code,
    repair_code_timed as openrouter_repair_code_timed,
)
from .scenarios import (
    SCENARIOS,
    get_all_scenarios,
    get_scenario_by_id,
    get_scenarios_by_tag,
    mock_patch,
    scenario_choices,
)
from .stats import (
    aggregate_session_stats,
    compute_quality_score,
    estimate_tokens,
    format_stats_line,
    heuristic_winner,
)

__all__ = [
    # database
    "init_db", "save_run", "get_runs", "get_run_by_id", "run_count",
    "delete_run", "get_leaderboard", "export_runs_csv", "export_runs_json",
    # diff_engine
    "compute_char_diff", "compute_line_diff", "render_html_diff",
    "diff_stats", "diff_summary",
    # ollama_client
    "is_ollama_running", "list_models", "model_exists", "generate",
    "repair_code", "repair_code_timed", "_strip_markdown_fences",
    # openrouter_client
    "is_configured", "openrouter_list_models",
    "openrouter_repair_code", "openrouter_repair_code_timed",
    # scenarios
    "SCENARIOS", "get_all_scenarios", "get_scenario_by_id",
    "get_scenarios_by_tag", "scenario_choices", "mock_patch",
    # stats
    "estimate_tokens", "compute_quality_score", "format_stats_line",
    "aggregate_session_stats", "heuristic_winner",
]
