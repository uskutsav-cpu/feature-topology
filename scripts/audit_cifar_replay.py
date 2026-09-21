"""Independently replay one immutable hosted CIFAR result on a fresh worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_remote_cifar import install_archive,validate_archive
from scripts.remote_cifar import parse_cell,sha256,verify_dataset
from scripts.run_cifar import CIFARResNet,ScaledModel,evaluate,load_data
from src.training.checkpoints import atomic_json,fingerprint


def replay_comparison(saved,replayed,samples):
    """Separate discrete prediction agreement from floating loss agreement."""
    saved_correct=round(float(saved["accuracy"])*samples)
    replayed_correct=round(float(replayed["accuracy"])*samples)
    return {
        "samples":samples,
        "saved_correct":saved_correct,
        "replayed_correct":replayed_correct,
        "correct_count_delta":replayed_correct-saved_correct,
        "loss_delta":float(replayed["loss"])-float(saved["loss"]),
        "accuracy_exact":saved_correct==replayed_correct,
        "loss_within_1e-4":abs(float(replayed["loss"])-float(saved["loss"]))<=1e-4,
    }


def audit(repo,cell,archive,status_path,expected_commit,output,source_specification):
    repo=Path(repo).resolve();cell=parse_cell(cell)
    if cell["stage"]!="production":
        raise ValueError("Replay audit requires a production cell")
    source_specification=Path(source_specification)
    if not source_specification.is_absolute():
        source_specification=repo/source_specification
    status=json.loads(Path(status_path).read_text())
    archived=validate_archive(archive,status,cell,source_specification,
                              expected_commit=expected_commit)
    if not archived.get("complete"):
        raise ValueError("Replay audit requires a complete immutable archive")
    install_archive(repo,archive,archived)
    config=archived["config"]
    run=repo/"results"/cell["dataset"].lower()/"runs"/fingerprint(config)
    metric_path=run/"metrics.json";checkpoint=run/"final.pt"
    representations=run/"representations.npz"
    metric=json.loads(metric_path.read_text())
    with np.load(representations,allow_pickle=False) as arrays:
        if any(not np.isfinite(arrays[name]).all() for name in arrays.files):
            raise ValueError("Nonfinite representation archive in replay audit")
    data=load_data(cell["dataset"],str(repo/"data/cifar"),download=True)
    _,dataset_sha256=verify_dataset(repo/"data/cifar",cell["dataset"],
                                    json.loads(source_specification.read_text()))
    torch.manual_seed(config["seed"])
    classes=10 if cell["dataset"]=="CIFAR10" else 100
    model=ScaledModel(CIFARResNet(classes),config["gamma"]).cpu()
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model.network.load_state_dict(state["network"])
    replayed=evaluate(model,data["test"],"cpu")
    saved=metric["test"]
    comparison=replay_comparison(saved,replayed,len(data["test"][0]))
    result={
        "schema":"feature-topology.cifar-hosted-replay.v1",
        "cell":cell,
        "run_id":archived["run_id"],
        "production_source_commit":expected_commit,
        "production_archive":Path(archive).name,
        "production_archive_sha256":sha256(archive),
        "production_status":Path(status_path).name,
        "production_status_sha256":sha256(status_path),
        "dataset_sha256":dataset_sha256,
        "source_specification_sha256":sha256(source_specification),
        "checkpoint_sha256":sha256(checkpoint),
        "metrics_sha256":sha256(metric_path),
        "representations_sha256":sha256(representations),
        "saved_test":saved,
        "replayed_test":replayed,
        "comparison":comparison,
        "host":{
            "platform":platform.platform(),"machine":platform.machine(),
            "python":platform.python_version(),"torch":torch.__version__,
            "commit":os.environ.get("GITHUB_SHA"),
            "workflow_run":os.environ.get("GITHUB_RUN_ID"),
        },
        "scope":"Independent source-platform replay; it does not alter the saved scientific metric.",
    }
    atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    parser.add_argument("--cell",required=True)
    parser.add_argument("--archive",required=True)
    parser.add_argument("--status",required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source-specification",
                        default="configs/completion/cifar_sources_v3.json")
    args=parser.parse_args()
    result=audit(args.repo,args.cell,args.archive,args.status,args.expected_commit,
                 args.output,args.source_specification)
    print(json.dumps({"run_id":result["run_id"],**result["comparison"]}))
    return int(not (result["comparison"]["accuracy_exact"]
                    and result["comparison"]["loss_within_1e-4"]))


if __name__=="__main__":
    raise SystemExit(main())
