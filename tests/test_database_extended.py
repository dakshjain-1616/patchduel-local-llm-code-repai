"""Tests for extended database functions (v2 columns, export, leaderboard, delete)."""
import csv
import json

import pytest

from patchduel_local_llm_.database import (
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


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test_extended.db")
    init_db(path)
    return path


def _save(db_path, model_a="m1", model_b="m2", winner=None, **kwargs):
    return save_run(model_a=model_a, model_b=model_b, buggy_code="x", winner=winner,
                    db_path=db_path, **kwargs)


# ── Migration / v2 columns ────────────────────────────────────────────────────

class TestV2Columns:
    def test_v2_columns_present_after_init(self, db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        conn.close()
        for col in ("duration_a", "duration_b", "tokens_a", "tokens_b",
                    "provider_a", "provider_b", "scenario_id"):
            assert col in cols

    def test_init_db_idempotent_with_v2(self, db_path):
        init_db(db_path)
        init_db(db_path)  # should not raise

    def test_v2_fields_stored_and_retrieved(self, db_path):
        run_id = save_run(
            model_a="llama3",
            model_b="mistral",
            buggy_code="x",
            duration_a=1.23,
            duration_b=4.56,
            tokens_a=100,
            tokens_b=200,
            provider_a="ollama",
            provider_b="openrouter",
            scenario_id="sc-001",
            db_path=db_path,
        )
        row = get_run_by_id(run_id, db_path)
        assert abs(row["duration_a"] - 1.23) < 0.001
        assert abs(row["duration_b"] - 4.56) < 0.001
        assert row["tokens_a"] == 100
        assert row["tokens_b"] == 200
        assert row["provider_a"] == "ollama"
        assert row["provider_b"] == "openrouter"
        assert row["scenario_id"] == "sc-001"

    def test_v2_fields_default_to_none(self, db_path):
        run_id = _save(db_path)
        row = get_run_by_id(run_id, db_path)
        assert row["duration_a"] is None
        assert row["tokens_a"] == 0 or row["tokens_a"] is None


# ── delete_run ────────────────────────────────────────────────────────────────

class TestDeleteRun:
    def test_delete_existing_run_returns_true(self, db_path):
        run_id = _save(db_path)
        assert delete_run(run_id, db_path) is True

    def test_deleted_run_not_found(self, db_path):
        run_id = _save(db_path)
        delete_run(run_id, db_path)
        assert get_run_by_id(run_id, db_path) is None

    def test_delete_nonexistent_returns_false(self, db_path):
        assert delete_run(9999, db_path) is False

    def test_delete_decrements_count(self, db_path):
        id1 = _save(db_path)
        id2 = _save(db_path)
        assert run_count(db_path) == 2
        delete_run(id1, db_path)
        assert run_count(db_path) == 1


# ── get_leaderboard ───────────────────────────────────────────────────────────

class TestGetLeaderboard:
    def test_empty_db_returns_empty_list(self, db_path):
        assert get_leaderboard(db_path) == []

    def test_returns_list_of_dicts(self, db_path):
        _save(db_path, model_a="llama3", model_b="mistral", winner="Model A")
        board = get_leaderboard(db_path)
        assert isinstance(board, list)
        assert isinstance(board[0], dict)

    def test_required_fields_present(self, db_path):
        _save(db_path, model_a="a", model_b="b", winner="Model A")
        board = get_leaderboard(db_path)
        for entry in board:
            for field in ("model", "runs", "wins", "ties", "losses", "win_rate"):
                assert field in entry

    def test_wins_counted_correctly(self, db_path):
        _save(db_path, model_a="llama3", model_b="mistral", winner="Model A")
        _save(db_path, model_a="llama3", model_b="mistral", winner="Model A")
        _save(db_path, model_a="llama3", model_b="mistral", winner="Model B")
        board = get_leaderboard(db_path)
        llama = next(e for e in board if e["model"] == "llama3")
        assert llama["wins"] == 2
        assert llama["losses"] == 1

    def test_sorted_by_wins_descending(self, db_path):
        _save(db_path, model_a="a", model_b="b", winner="Model A")
        _save(db_path, model_a="a", model_b="b", winner="Model A")
        _save(db_path, model_a="a", model_b="b", winner="Model B")
        board = get_leaderboard(db_path)
        wins = [e["wins"] for e in board]
        assert wins == sorted(wins, reverse=True)

    def test_tie_counted(self, db_path):
        _save(db_path, model_a="x", model_b="y", winner="Tie")
        board = get_leaderboard(db_path)
        x = next(e for e in board if e["model"] == "x")
        assert x["ties"] == 1


# ── export_runs_csv ───────────────────────────────────────────────────────────

class TestExportRunsCsv:
    def test_empty_db_returns_empty_string(self, db_path):
        assert export_runs_csv(db_path) == ""

    def test_returns_valid_csv(self, db_path):
        _save(db_path, model_a="llama3", model_b="mistral", winner="Model A")
        csv_text = export_runs_csv(db_path)
        reader = list(csv.DictReader(csv_text.splitlines()))
        assert len(reader) == 1
        assert reader[0]["model_a"] == "llama3"
        assert reader[0]["winner"] == "Model A"

    def test_multiple_rows(self, db_path):
        for _ in range(5):
            _save(db_path)
        csv_text = export_runs_csv(db_path)
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert len(rows) == 5

    def test_has_header_row(self, db_path):
        _save(db_path)
        csv_text = export_runs_csv(db_path)
        first_line = csv_text.splitlines()[0]
        assert "model_a" in first_line
        assert "timestamp" in first_line


# ── export_runs_json ──────────────────────────────────────────────────────────

class TestExportRunsJson:
    def test_empty_db_returns_empty_array(self, db_path):
        result = json.loads(export_runs_json(db_path))
        assert result == []

    def test_returns_valid_json(self, db_path):
        _save(db_path, model_a="llama3", model_b="mistral")
        data = json.loads(export_runs_json(db_path))
        assert isinstance(data, list)
        assert data[0]["model_a"] == "llama3"

    def test_diff_html_excluded(self, db_path):
        save_run(
            model_a="a", model_b="b", buggy_code="x",
            diff_html_a="<div>BIG HTML</div>",
            diff_html_b="<div>OTHER</div>",
            db_path=db_path,
        )
        data = json.loads(export_runs_json(db_path))
        assert "diff_html_a" not in data[0]
        assert "diff_html_b" not in data[0]

    def test_multiple_runs(self, db_path):
        for i in range(3):
            _save(db_path, model_a=f"m{i}", model_b="m_other")
        data = json.loads(export_runs_json(db_path))
        assert len(data) == 3
