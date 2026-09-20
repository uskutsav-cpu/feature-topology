"""Drive hosted frozen-v3 CIFAR calibration and production to completion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_remote_cifar import collect
from scripts.remote_cifar import GAMMAS, MULTIPLIERS, cell_id, config_for, parse_cell
from src.training.checkpoints import atomic_json, fingerprint


def run(command: list[str]) -> str:
    return subprocess.run(command,check=True,text=True,capture_output=True).stdout.strip()


def calibration_cells(dataset: str) -> list[dict]:
    return [parse_cell({"stage":"calibration","dataset":dataset,"gamma":gamma,
                        "multiplier":multiplier})
            for gamma in GAMMAS for multiplier in MULTIPLIERS]


def finalize_calibration(repo: Path, dataset: str) -> dict:
    root=repo/"results"/dataset.lower()
    chosen={}
    for gamma in GAMMAS:
        candidates=[]
        for multiplier in MULTIPLIERS:
            cell={"stage":"calibration","dataset":dataset,"gamma":gamma,"multiplier":multiplier}
            config=config_for(cell)
            directory=root/"calibration"
            summary=json.loads((directory/fingerprint(config)/"summary.json").read_text())
            if summary["status"]!="diverged":
                last=summary["history"][-1]
                candidates.append({"lr":config["lr"],"reached":summary["status"]=="converged",
                                   "loss":last["training_loss"],"step":last["step"]})
        if not candidates:
            raise RuntimeError(f"No stable hosted CIFAR rate: {dataset} gamma={gamma}")
        chosen[str(gamma)]=min(candidates,key=lambda row:(not row["reached"],
                                  row["step"] if row["reached"] else row["loss"]))
    value={"selection":chosen,"calibration_steps":2000,
           "protocol":{"classes":10 if dataset=="CIFAR10" else 100,"dataset":dataset,
                       "batch_size":128,"target_loss":.2,"eval_every":500},
           "device":"cpu-hosted"}
    target=root/"gamma_to_lr.json"
    if target.exists() and json.loads(target.read_text())!=value:
        raise ValueError(f"Conflicting frozen CIFAR calibration map: {target}")
    atomic_json(target,value)
    return value


def production_cells(dataset: str, frozen: dict) -> list[dict]:
    return [parse_cell({"stage":"production","dataset":dataset,"gamma":gamma,
                        "seed":seed,"lr":frozen["selection"][str(gamma)]["lr"]})
            for gamma in GAMMAS for seed in range(5)]


def workflow_state(run_id: str, retries: int = 6, initial_delay: float = 2) -> dict:
    """Read workflow state with bounded retries for transient API failures.

    Only this idempotent read is retried.  Workflow dispatch remains a
    single-attempt operation so an ambiguous response cannot duplicate cells.
    """
    for attempt in range(retries):
        try:
            payload=run(["gh","run","view",run_id,"--json",
                         "status,conclusion,jobs,url"])
            break
        except subprocess.CalledProcessError:
            if attempt + 1 == retries:
                raise
            time.sleep(min(initial_delay * 2**attempt,30))
    value=json.loads(payload)
    counts={}
    for job in value["jobs"]:
        counts[job["status"]]=counts.get(job["status"],0)+1
    return {"status":value["status"],"conclusion":value["conclusion"],
            "counts":counts,"url":value["url"]}


def dispatch(cells: list[dict], tag: str) -> tuple[str,str]:
    payload=json.dumps(cells,separators=(",",":"))
    output=run(["gh","workflow","run","cifar_v3.yml","--ref",tag,
                "-f",f"cells={payload}","-f",f"result_tag={tag}"])
    match=re.search(r"/actions/runs/(\d+)",output)
    if not match:
        raise ValueError(f"Could not parse CIFAR workflow URL: {output}")
    return match.group(1),output


def ensure_release(tag: str, commit: str) -> None:
    result=subprocess.run(["gh","release","view",tag],text=True,capture_output=True)
    if result.returncode:
        run(["gh","release","create",tag,"--target",commit,"--prerelease",
             "--title",f"Frozen v3 hosted CIFAR work: {tag}",
             "--notes","Immutable in-progress hosted artifacts for the frozen v3 CIFAR study; not a final research release."])
    remote=run(["git","ls-remote","origin",f"refs/tags/{tag}"]).split()
    if not remote or remote[0]!=commit:
        raise ValueError(f"CIFAR result tag is not bound to source commit {commit}")


def resume_unfinished_attempts(state: dict, state_path: Path, poll_seconds: int) -> None:
    """Finish controller bookkeeping after an interruption without redispatch."""
    for attempt in state.get("attempts",[]):
        if attempt.get("finished"):
            continue
        previous=None
        while True:
            current=workflow_state(attempt["workflow_run"])
            if current!=previous:
                print(json.dumps({"workflow_run":attempt["workflow_run"],**current}),flush=True)
                previous=current
            if current["status"]=="completed":
                attempt.update(finished=time.time(),conclusion=current["conclusion"])
                atomic_json(state_path,state)
                break
            time.sleep(poll_seconds)


def finish_stage(repo: Path, dataset: str, stage: str, cells: list[dict], tag: str,
                 state: dict, state_path: Path, poll_seconds: int, max_stalled: int,
                 cache_root: Path) -> None:
    cache=cache_root/tag
    stalled=0
    while True:
        report=collect(repo,tag,cells,cache,repo/"configs/completion/cifar_sources_v3.json",
                       install=True,expected_commit=state["source_commit"])
        atomic_json(repo/"results/completion/remote_collections"/f"{dataset.lower()}_{stage}.json",report)
        state[dataset][stage]={"expected":report["expected"],"complete":len(report["complete"]),
                               "missing":len(report["missing"]),"invalid":report["invalid"]}
        atomic_json(state_path,state)
        print(json.dumps({"dataset":dataset,"stage":stage,"complete":len(report["complete"]),
                          "expected":report["expected"],"invalid":len(report["invalid"])}),flush=True)
        if report["invalid"]:
            raise RuntimeError(f"Invalid hosted CIFAR artifacts: {dataset}/{stage}")
        if not report["missing"]:
            return
        before=len(report["complete"])
        missing=set(report["missing"])
        batch=[cell for cell in cells if cell_id(cell) in missing][:64]
        run_id,url=dispatch(batch,tag)
        attempt={"dataset":dataset,"stage":stage,"workflow_run":run_id,"url":url,
                 "cell_ids":[cell_id(cell) for cell in batch],"started":time.time()}
        state["attempts"].append(attempt);atomic_json(state_path,state)
        previous=None
        while True:
            current=workflow_state(run_id)
            if current!=previous:
                print(json.dumps({"workflow_run":run_id,**current}),flush=True);previous=current
            if current["status"]=="completed":
                attempt.update(finished=time.time(),conclusion=current["conclusion"])
                atomic_json(state_path,state);break
            time.sleep(poll_seconds)
        after=collect(repo,tag,cells,cache,repo/"configs/completion/cifar_sources_v3.json",
                      install=True,expected_commit=state["source_commit"])
        stalled=stalled+1 if len(after["complete"])<=before else 0
        if stalled>=max_stalled:
            raise RuntimeError(f"No verified hosted CIFAR progress: {dataset}/{stage}")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    parser.add_argument("--dataset",choices=["CIFAR10","CIFAR100"],required=True)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--state",default="results/completion/cifar_remote_queue.json")
    parser.add_argument("--poll-seconds",type=int,default=30)
    parser.add_argument("--max-stalled-attempts",type=int,default=3)
    parser.add_argument("--cache",default="results/completion/remote_cifar_cache")
    args=parser.parse_args();repo=Path(args.repo).resolve();state_path=repo/args.state
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        "schema":"feature-topology.remote-cifar-queue.v1","attempts":[]}
    commit=state.get("source_commit") or run(["git","rev-parse","HEAD"])
    state["source_commit"]=commit
    state.setdefault(args.dataset,{})
    ensure_release(args.tag,commit)
    resume_unfinished_attempts(state,state_path,args.poll_seconds)
    cache_root=Path(args.cache)
    if not cache_root.is_absolute():
        cache_root=repo/cache_root
    calibration=calibration_cells(args.dataset)
    finish_stage(repo,args.dataset,"calibration",calibration,args.tag,state,state_path,
                 args.poll_seconds,args.max_stalled_attempts,cache_root)
    frozen=finalize_calibration(repo,args.dataset)
    production=production_cells(args.dataset,frozen)
    finish_stage(repo,args.dataset,"production",production,args.tag,state,state_path,
                 args.poll_seconds,args.max_stalled_attempts,cache_root)
    print(json.dumps({"status":"complete","dataset":args.dataset,"runs":len(production)}),flush=True)


if __name__=="__main__":
    main()
