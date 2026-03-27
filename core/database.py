"""SQLite persistence for PatchDuel evaluation runs."""
import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "patchduel.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    model_a     TEXT    NOT NULL,
    model_b     TEXT    NOT NULL,
    buggy_code  TEXT    NOT NULL,
    patch_a     TEXT,
    patch_b     TEXT,
    diff_html_a TEXT,
    diff_html_b TEXT,
    winner      TEXT,
    notes       TEXT
);
"""

# Columns added in schema v2 — applied via safe migration
_V2_COLUMNS: list[tuple[str, str]] = [
    ("duration_a", "REAL"),
    ("duration_b", "REAL"),
    ("tokens_a", "INTEGER"),
    ("tokens_b", "INTEGER"),
    ("provider_a", "TEXT"),
    ("provider_b", "TEXT"),
    ("scenario_id", "TEXT"),
]


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory enabled."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add v2 columns to an existing runs table without touching existing rows."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col_name, col_type in _V2_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")


def init_db(db_path: str | None = None) -> None:
    """Create the runs table (if not exists) and apply any pending migrations."""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        _migrate(conn)
        conn.commit()


def save_run(
    model_a: str,
    model_b: str,
    buggy_code: str,
    patch_a: str = "",
    patch_b: str = "",
    diff_html_a: str = "",
    diff_html_b: str = "",
    winner: str | None = None,
    notes: str | None = None,
    # v2 fields (optional — default to None / 0)
    duration_a: float | None = None,
    duration_b: float | None = None,
    tokens_a: int = 0,
    tokens_b: int = 0,
    provider_a: str = "ollama",
    provider_b: str = "ollama",
    scenario_id: str | None = None,
    db_path: str | None = None,
) -> int:
    """
    Persist a comparison run to the database.
    Returns the new row id.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs
              (timestamp, model_a, model_b, buggy_code,
               patch_a, patch_b, diff_html_a, diff_html_b, winner, notes,
               duration_a, duration_b, tokens_a, tokens_b,
               provider_a, provider_b, scenario_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp, model_a, model_b, buggy_code,
                patch_a, patch_b, diff_html_a, diff_html_b, winner, notes,
                duration_a, duration_b, tokens_a, tokens_b,
                provider_a, provider_b, scenario_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


def get_runs(limit: int = 50, db_path: str | None = None) -> list[dict]:
    """Return the most recent `limit` runs, newest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_run_by_id(run_id: int, db_path: str | None = None) -> dict | None:
    """Fetch a single run by its primary key."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def run_count(db_path: str | None = None) -> int:
    """Return total number of saved runs."""
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def delete_run(run_id: int, db_path: str | None = None) -> bool:
    """Delete a run by ID. Returns True if a row was deleted."""
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_leaderboard(db_path: str | None = None) -> list[dict]:
    """
    Return per-model win/loss/tie statistics across all saved runs.
    Sorted by win count descending.
    """
    runs = get_runs(limit=10_000, db_path=db_path)
    model_stats: dict[str, dict] = {}

    for run in runs:
        for model, role in [(run["model_a"], "A"), (run["model_b"], "B")]:
            if not model:
                continue
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "runs": 0,
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                }
            s = model_stats[model]
            s["runs"] += 1
            winner = run.get("winner") or ""
            if winner == f"Model {role}":
                s["wins"] += 1
            elif winner == "Tie":
                s["ties"] += 1
            elif winner in ("Model A", "Model B"):
                s["losses"] += 1

    result = sorted(model_stats.values(), key=lambda x: (-x["wins"], -x["runs"]))
    for s in result:
        s["win_rate"] = f"{s['wins'] / s['runs']:.0%}" if s["runs"] else "—"
    return result


def export_runs_csv(db_path: str | None = None) -> str:
    """Serialize all runs to a CSV string."""
    runs = get_runs(limit=10_000, db_path=db_path)
    if not runs:
        return ""

    _FIELDS = [
        "id", "timestamp", "model_a", "model_b", "winner", "notes",
        "duration_a", "duration_b", "tokens_a", "tokens_b",
        "provider_a", "provider_b", "scenario_id", "buggy_code",
        "patch_a", "patch_b",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for run in runs:
        writer.writerow({k: run.get(k, "") for k in _FIELDS})
    return buf.getvalue()


def export_runs_json(db_path: str | None = None) -> str:
    """Serialize all runs to a JSON string (excluding large diff HTML blobs)."""
    runs = get_runs(limit=10_000, db_path=db_path)
    _EXCLUDE = {"diff_html_a", "diff_html_b"}
    trimmed = [{k: v for k, v in r.items() if k not in _EXCLUDE} for r in runs]
    return json.dumps(trimmed, indent=2, default=str)
