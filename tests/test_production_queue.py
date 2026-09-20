import json
import subprocess
import pytest
from scripts import run_production_queue
from scripts.run_production_queue import begin_attempt, validate_registry


def write_index(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": [{"run_id": value} for value in ids]}))


def test_registry_accounts_unique_profiles_and_shared_primary(tmp_path):
    write_index(tmp_path / "primary.json", ["a", "b"])
    write_index(tmp_path / "width.json", ["a", "c", "d"])
    registry = {"shared_primary_index": "primary.json", "unique_profile_total": 4,
                "cohorts": [
                    {"name": "primary", "index": "primary.json", "expected_unique_profiles": 2},
                    {"name": "width", "index": "width.json", "exclude_shared_primary": True,
                     "expected_unique_profiles": 2}]}
    assert validate_registry(tmp_path, registry) == {"a", "b"}
    registry["cohorts"][1]["exclude_shared_primary"] = False
    with pytest.raises(ValueError, match="count mismatch"):
        validate_registry(tmp_path, registry)


def test_registry_rejects_cross_cohort_duplication(tmp_path):
    write_index(tmp_path / "primary.json", ["a"])
    write_index(tmp_path / "one.json", ["b"])
    write_index(tmp_path / "two.json", ["b"])
    registry = {"shared_primary_index": "primary.json", "unique_profile_total": 3,
                "cohorts": [
                    {"name": "primary", "index": "primary.json", "expected_unique_profiles": 1},
                    {"name": "one", "index": "one.json", "expected_unique_profiles": 1},
                    {"name": "two", "index": "two.json", "expected_unique_profiles": 1}]}
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry(tmp_path, registry)


def test_adoption_resumes_existing_unfinished_attempt():
    original = {"cohort": "width", "workflow_run": "123", "url": "old",
                "run_ids": ["a", "b"], "started": 1.0}
    state = {"attempts": [original]}
    resumed = begin_attempt(
        state, "width", "123", "new", ["b", "a"], adopted=True
    )
    assert resumed is original
    assert resumed["url"] == "new"
    assert len(state["attempts"]) == 1


def test_adoption_rejects_changed_or_final_attempt():
    state = {"attempts": [{"cohort": "width", "workflow_run": "123",
                            "url": "old", "run_ids": ["a"], "started": 1.0}]}
    with pytest.raises(ValueError, match="run IDs changed"):
        begin_attempt(state, "width", "123", "new", ["b"], adopted=True)
    state["attempts"][0]["finished"] = 2.0
    with pytest.raises(ValueError, match="already-final"):
        begin_attempt(state, "width", "123", "new", ["a"], adopted=True)


def test_status_poll_retries_transient_read_only_failure(monkeypatch):
    calls=[]
    def fake_run(command, capture=True):
        calls.append(command)
        if len(calls)==1:
            raise subprocess.CalledProcessError(1,command)
        return json.dumps({"status":"queued","conclusion":"","url":"https://example/run",
                           "jobs":[{"name":"metrics (0123456789abcdef)",
                                    "status":"in_progress"}]})
    monkeypatch.setattr(run_production_queue,"run",fake_run)
    monkeypatch.setattr(run_production_queue.time,"sleep",lambda _:None)
    state=run_production_queue.workflow_state("123",retries=2,initial_delay=0)
    assert len(calls)==2
    assert state["counts"]=={"in_progress":1}
    assert state["run_ids"]==["0123456789abcdef"]


def test_status_poll_exhausts_bounded_retries(monkeypatch):
    def fail(command, capture=True):
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(run_production_queue,"run",fail)
    monkeypatch.setattr(run_production_queue.time,"sleep",lambda _:None)
    with pytest.raises(subprocess.CalledProcessError):
        run_production_queue.workflow_state("123",retries=2,initial_delay=0)
