"""Run the checksum-bound frozen-v3 hosted metric queue to completion."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_remote_metrics import collect
from src.training.checkpoints import atomic_json


def run(command, *, capture=True):
    return subprocess.run(command, check=True, text=True,
                          capture_output=capture).stdout.strip()


def validate_registry(repo, registry):
    primary = json.loads((repo / registry["shared_primary_index"]).read_text())
    primary_ids = {row["run_id"] for row in primary["runs"]}
    seen = set()
    for cohort in registry["cohorts"]:
        index = json.loads((repo / cohort["index"]).read_text())
        ids = {row["run_id"] for row in index["runs"]}
        if cohort.get("exclude_shared_primary"):
            ids -= primary_ids
        if len(ids) != cohort["expected_unique_profiles"]:
            raise ValueError(f"Registry count mismatch for {cohort['name']}")
        overlap = ids & seen
        if overlap:
            raise ValueError(f"Registry would duplicate {len(overlap)} profiles in {cohort['name']}")
        seen.update(ids)
    if len(seen) != registry["unique_profile_total"]:
        raise ValueError("Registry unique-profile total mismatch")
    return primary_ids


def dispatch(registry, cohort, run_ids):
    payload = json.dumps(run_ids, separators=(",", ":"))
    output = run(["gh", "workflow", "run", registry["workflow"],
                  "--ref", registry["workflow_ref"],
                  "-f", f"index_path={cohort['index']}",
                  "-f", f"run_ids={payload}",
                  "-f", f"input_tag={cohort['input_tag']}",
                  "-f", f"result_tag={cohort['result_tag']}"])
    match = re.search(r"/actions/runs/(\d+)", output)
    if not match:
        raise ValueError(f"Could not parse dispatched workflow URL: {output}")
    return match.group(1), output


def workflow_state(run_id):
    payload = run(["gh", "run", "view", str(run_id), "--json", "status,conclusion,jobs,url"])
    value = json.loads(payload)
    counts = {}
    run_ids = []
    for job in value["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
        match = re.fullmatch(r"metrics \(([0-9a-f]{16})\)", job["name"])
        if match:
            run_ids.append(match.group(1))
    return {"status": value["status"], "conclusion": value["conclusion"],
            "counts": counts, "url": value["url"], "run_ids": sorted(run_ids)}


def begin_attempt(state, cohort, run_id, url, run_ids, adopted=False):
    """Create a workflow ledger row, or resume its interrupted controller."""
    if adopted:
        matches = [row for row in state["attempts"]
                   if row["cohort"] == cohort and row["workflow_run"] == run_id]
        if matches:
            attempt = matches[-1]
            if attempt.get("finished"):
                raise ValueError(f"Cannot adopt already-final workflow {run_id}")
            if sorted(attempt["run_ids"]) != sorted(run_ids):
                raise ValueError(f"Adopted workflow run IDs changed: {run_id}")
            attempt["url"] = url
            return attempt
    attempt = {"cohort": cohort, "workflow_run": run_id,
               "url": url, "run_ids": run_ids, "started": time.time()}
    state["attempts"].append(attempt)
    return attempt


def main(args):
    repo = Path(args.repo).resolve()
    registry = json.loads((repo / args.registry).read_text())
    primary_ids = validate_registry(repo, registry)
    cohorts=registry["cohorts"]
    names=[cohort["name"] for cohort in cohorts]
    if args.start_at:
        if args.start_at not in names:
            raise ValueError(f"Unknown start cohort: {args.start_at}")
        cohorts=cohorts[names.index(args.start_at):]
    if args.stop_after:
        selected_names=[cohort["name"] for cohort in cohorts]
        if args.stop_after not in selected_names:
            raise ValueError(f"Unknown stop cohort: {args.stop_after}")
        cohorts=cohorts[:selected_names.index(args.stop_after)+1]
    state_path = repo / args.state
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "feature-topology.production-queue-state.v1", "attempts": [], "cohorts": {}
    }
    adopted = args.adopt_run
    for cohort in cohorts:
        excluded = primary_ids if cohort.get("exclude_shared_primary") else set()
        retries_without_progress = 0
        while True:
            report_path = repo / "results" / "completion" / "remote_collections" / f"{cohort['name']}.json"
            report = collect(repo, repo / cohort["index"], cohort["input_tag"], cohort["result_tag"],
                             repo / args.cache, install=True, excluded=excluded)
            atomic_json(report_path, report)
            complete = len(report["complete"])
            state["cohorts"][cohort["name"]] = {
                "expected": report["expected"], "complete": complete,
                "missing": len(report["missing"]), "invalid": report["invalid"]
            }
            atomic_json(state_path, state)
            print(json.dumps({"cohort": cohort["name"], "complete": complete,
                              "expected": report["expected"], "invalid": len(report["invalid"])}), flush=True)
            if report["invalid"]:
                raise RuntimeError(f"Invalid hosted artifacts in {cohort['name']}")
            if not report["missing"]:
                break
            before = complete
            batch = report["missing"][:24]
            if adopted:
                adopted_state = workflow_state(adopted)
                run_id, url, batch = adopted, adopted_state["url"], adopted_state["run_ids"]
                adopted = None
                was_adopted = True
            else:
                run_id, url = dispatch(registry, cohort, batch)
                was_adopted = False
            attempt = begin_attempt(
                state, cohort["name"], run_id, url, batch, adopted=was_adopted
            )
            atomic_json(state_path, state)
            previous = None
            while True:
                current = workflow_state(run_id)
                if current != previous:
                    print(json.dumps({"workflow_run": run_id, **current}), flush=True)
                    previous = current
                if current["status"] == "completed":
                    attempt.update(finished=time.time(), conclusion=current["conclusion"])
                    atomic_json(state_path, state)
                    break
                time.sleep(args.poll_seconds)
            # Collection on the next loop is the authoritative success check;
            # a failed workflow may still have uploaded compatible partial work.
            after = collect(repo, repo / cohort["index"], cohort["input_tag"], cohort["result_tag"],
                            repo / args.cache, install=True, excluded=excluded)
            retries_without_progress = retries_without_progress + 1 if len(after["complete"]) <= before else 0
            if retries_without_progress >= args.max_stalled_attempts:
                raise RuntimeError(f"No verified progress in {cohort['name']} after repeated attempts")
    print(json.dumps({"status": "complete", "unique_profiles": registry["unique_profile_total"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--registry", default="configs/completion/production_cohorts_v3.json")
    parser.add_argument("--state", default="results/completion/production_queue_v3.json")
    parser.add_argument("--cache", default="results/completion/remote_metric_cache")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-stalled-attempts", type=int, default=3)
    parser.add_argument("--adopt-run", help="Existing workflow run for the first incomplete cohort")
    parser.add_argument("--start-at", help="Start at this registry cohort (inclusive)")
    parser.add_argument("--stop-after", help="Stop after this registry cohort (inclusive)")
    main(parser.parse_args())
