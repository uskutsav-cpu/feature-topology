"""Collect immutable successful source-platform CIFAR replay audit reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.remote_cifar import sha256
from src.training.checkpoints import atomic_json


TAG=re.compile(r"[0-9A-Za-z._-]+")
REPORT=re.compile(r"replay-([0-9a-f]{16})-([0-9]+)\.json")


def validate_report(path,audit_commit):
    path=Path(path);match=REPORT.fullmatch(path.name)
    value=json.loads(path.read_text())
    comparison=value.get("comparison",{})
    if (match is None or value.get("schema")!="feature-topology.cifar-hosted-replay.v1"
            or value.get("run_id")!=match.group(1)
            or value.get("host",{}).get("commit")!=audit_commit
            or value.get("host",{}).get("workflow_run")!=match.group(2)
            or not comparison.get("accuracy_exact")
            or not comparison.get("loss_within_1e-4")):
        raise ValueError(f"Invalid hosted CIFAR replay report: {path.name}")
    return value


def collect(repo,tag,audit_commit,directory,output):
    repo=Path(repo).resolve();directory=Path(directory)
    if not directory.is_absolute(): directory=repo/directory
    output=Path(output)
    if not output.is_absolute(): output=repo/output
    if not TAG.fullmatch(tag) or not re.fullmatch(r"[0-9a-f]{40}",audit_commit):
        raise ValueError("Unsafe replay release identity")
    directory.mkdir(parents=True,exist_ok=True)
    subprocess.run(["gh","release","download",tag,"--dir",str(directory),
                    "--skip-existing","--pattern","replay-*.json"],check=True)
    records=[];seen=set()
    for path in sorted(directory.glob("replay-*.json")):
        value=validate_report(path,audit_commit)
        if value["run_id"] in seen:
            raise ValueError(f"Duplicate hosted replay: {value['run_id']}")
        seen.add(value["run_id"])
        records.append({"run_id":value["run_id"],
                        "path":path.relative_to(repo).as_posix(),"sha256":sha256(path)})
    if not records:
        raise ValueError("No successful hosted CIFAR replay report found")
    result={"schema":"feature-topology.cifar-hosted-replays.v1",
            "audit_tag":tag,"audit_commit":audit_commit,"records":records}
    atomic_json(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    parser.add_argument("--tag",required=True)
    parser.add_argument("--audit-commit",required=True)
    parser.add_argument("--directory",default="results/completion/cifar_hosted_replays")
    parser.add_argument("--output",default="results/completion/cifar_hosted_replays_v3.json")
    args=parser.parse_args()
    result=collect(args.repo,args.tag,args.audit_commit,args.directory,args.output)
    print(json.dumps({"reports":len(result["records"]),"tag":result["audit_tag"]}))
