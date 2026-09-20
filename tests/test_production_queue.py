import json
import pytest
from scripts.run_production_queue import validate_registry


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
