"""Tests for the SQLite database layer."""
import os
import tempfile
from datetime import datetime, timezone

import pytest

from patchduel_local_llm_.database import (
    get_run_by_id,
    get_runs,
    init_db,
    run_count,
    save_run,
)


@pytest.fixture()
def db_path(tmp_path):
    """Provide a fresh temporary SQLite database for each test."""
    path = str(tmp_path / "test_patchduel.db")
    init_db(path)
    return path


class TestInitDb:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "new.db")
        init_db(path)
        assert os.path.exists(path)

    def test_idempotent(self, db_path):
        # Calling init_db twice should not raise
        init_db(db_path)
        init_db(db_path)


class TestSaveRun:
    """Test spec case 2: 'Click Save Run → SQLite DB row created with timestamp/model_id'."""

    def test_returns_integer_id(self, db_path):
        run_id = save_run(
            model_a="llama3.2",
            model_b="mistral-nemo",
            buggy_code="def add(a,b): return a-b",
            db_path=db_path,
        )
        assert isinstance(run_id, int)
        assert run_id >= 1

    def test_row_exists_after_save(self, db_path):
        run_id = save_run(
            model_a="llama3.2",
            model_b="mistral-nemo",
            buggy_code="def add(a,b): return a-b",
            db_path=db_path,
        )
        row = get_run_by_id(run_id, db_path)
        assert row is not None

    def test_model_id_stored_correctly(self, db_path):
        save_run(
            model_a="llama3.2",
            model_b="mistral-nemo",
            buggy_code="x = 1",
            db_path=db_path,
        )
        rows = get_runs(1, db_path)
        assert rows[0]["model_a"] == "llama3.2"
        assert rows[0]["model_b"] == "mistral-nemo"

    def test_timestamp_is_iso_format(self, db_path):
        save_run("m1", "m2", "code", db_path=db_path)
        rows = get_runs(1, db_path)
        ts = rows[0]["timestamp"]
        # Should parse without error
        dt = datetime.fromisoformat(ts)
        assert dt.year >= 2024

    def test_all_fields_stored(self, db_path):
        run_id = save_run(
            model_a="llama3.2",
            model_b="mistral-nemo",
            buggy_code="buggy",
            patch_a="fixed_a",
            patch_b="fixed_b",
            diff_html_a="<div>A</div>",
            diff_html_b="<div>B</div>",
            winner="Model A",
            notes="test run",
            db_path=db_path,
        )
        row = get_run_by_id(run_id, db_path)
        assert row["buggy_code"] == "buggy"
        assert row["patch_a"] == "fixed_a"
        assert row["patch_b"] == "fixed_b"
        assert row["diff_html_a"] == "<div>A</div>"
        assert row["winner"] == "Model A"
        assert row["notes"] == "test run"

    def test_multiple_runs_increments_id(self, db_path):
        id1 = save_run("m1", "m2", "code1", db_path=db_path)
        id2 = save_run("m1", "m2", "code2", db_path=db_path)
        assert id2 > id1

    def test_run_count_increments(self, db_path):
        assert run_count(db_path) == 0
        save_run("m1", "m2", "c1", db_path=db_path)
        assert run_count(db_path) == 1
        save_run("m1", "m2", "c2", db_path=db_path)
        assert run_count(db_path) == 2


class TestGetRuns:
    def test_empty_db_returns_empty_list(self, db_path):
        assert get_runs(10, db_path) == []

    def test_returns_list_of_dicts(self, db_path):
        save_run("m1", "m2", "code", db_path=db_path)
        rows = get_runs(10, db_path)
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)

    def test_newest_first_ordering(self, db_path):
        id1 = save_run("m1", "m2", "c1", db_path=db_path)
        id2 = save_run("m1", "m2", "c2", db_path=db_path)
        rows = get_runs(10, db_path)
        # Newest first → id2 should appear before id1
        ids = [r["id"] for r in rows]
        assert ids.index(id2) < ids.index(id1)

    def test_limit_respected(self, db_path):
        for i in range(10):
            save_run("m1", "m2", f"code{i}", db_path=db_path)
        assert len(get_runs(5, db_path)) == 5

    def test_get_run_by_id_not_found(self, db_path):
        assert get_run_by_id(9999, db_path) is None
