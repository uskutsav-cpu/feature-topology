import json
from pathlib import Path
import tarfile

import pytest

from scripts import collect_remote_cifar, remote_cifar
from src.training.checkpoints import atomic_json


def calibration_cell(**updates):
    value={"stage":"calibration","dataset":"CIFAR10","gamma":.125,"multiplier":.125}
    value.update(updates)
    return value


def test_cells_reconstruct_frozen_original_configs():
    cell=calibration_cell()
    config=remote_cifar.config_for(cell)
    assert config == {"classes":10,"dataset":"CIFAR10","batch_size":128,
                      "target_loss":.2,"eval_every":500,"gamma":.125,
                      "lr":.1*.125**2*.125,"seed":900,"max_steps":2000}
    production={"stage":"production","dataset":"CIFAR100","gamma":128.,"seed":4,"lr":.03}
    assert remote_cifar.config_for(production) == {
        "classes":100,"dataset":"CIFAR100","batch_size":128,"target_loss":.2,
        "eval_every":500,"gamma":128.,"seed":4,"lr":.03,"max_steps":35200}


@pytest.mark.parametrize("cell",[
    {}, calibration_cell(gamma=.3), calibration_cell(multiplier=3),
    {"stage":"production","dataset":"CIFAR10","gamma":.125,"seed":5,"lr":.1},
    {"stage":"production","dataset":"CIFAR10","gamma":.125,"seed":0,"lr":float("nan")},
])
def test_invalid_cells_fail_closed(cell):
    with pytest.raises(ValueError): remote_cifar.parse_cell(cell)


def test_plan_rejects_duplicates_and_oversize():
    cell=calibration_cell()
    with pytest.raises(ValueError,match="unique"):
        remote_cifar.plan(json.dumps([cell,cell]))
    with pytest.raises(ValueError,match="1--64"):
        remote_cifar.plan(json.dumps([dict(cell,multiplier=.125) for _ in range(65)]))


def test_pack_and_restore_checksum_bound_resume(tmp_path):
    cell=calibration_cell()
    work=tmp_path/"work"; data=tmp_path/"data"; outgoing=tmp_path/"outgoing"
    source=tmp_path/"sources.json"
    archive=data/"fixture.tar.gz"; archive.parent.mkdir(); archive.write_bytes(b"dataset")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":archive.name,"bytes":len(b"dataset"),
                   "sha256":remote_cifar.sha256(archive)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    config=remote_cifar.config_for(cell)
    atomic_json(directory/"config.json",config)
    atomic_json(directory/"summary.json",{
        "config":config,"run_id":directory.name,"status":"budget_exhausted","history":[]})
    (directory/"final.pt").write_bytes(b"checkpoint")
    (directory/"resume.pt").write_bytes(b"terminal optimizer state")
    report=remote_cifar.pack(work,data,outgoing,cell,"job-1",source)
    assert report["complete"]
    saved=next(outgoing.glob("*.tar.gz"))
    resume=tmp_path/"resume"; resume.mkdir(); target=resume/saved.name; target.write_bytes(saved.read_bytes())
    restored=tmp_path/"restored"
    assert remote_cifar.restore_archives(resume,restored,cell,remote_cifar.sha256(archive))==3
    assert not any(name.endswith("resume.pt") for name in report["files"])
    assert (remote_cifar.run_directory(restored,cell)/"final.pt").read_bytes()==b"checkpoint"
    with tarfile.open(target,"r:gz") as tar:
        assert "job_manifest.json" in tar.getnames()


def test_pack_partial_checkpoint_is_resumable(tmp_path):
    cell=calibration_cell(); work=tmp_path/"work"; data=tmp_path/"data"
    source=tmp_path/"sources.json"; archive=data/"fixture"; archive.parent.mkdir(); archive.write_bytes(b"x")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":archive.name,"bytes":1,"sha256":remote_cifar.sha256(archive)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    (directory/"resume.pt").write_bytes(b"partial")
    report=remote_cifar.pack(work,data,tmp_path/"out",cell,"job-2",source)
    assert not report["complete"] and any(name.endswith("resume.pt") for name in report["files"])


def test_restore_uses_newest_cumulative_attempt(tmp_path):
    cell=calibration_cell(); work=tmp_path/"work"; data=tmp_path/"data"; out=tmp_path/"out"
    source=tmp_path/"sources.json"; archive=data/"fixture"; archive.parent.mkdir(); archive.write_bytes(b"x")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":archive.name,"bytes":1,"sha256":remote_cifar.sha256(archive)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    (directory/"resume.pt").write_bytes(b"older")
    remote_cifar.pack(work,data,out,cell,"0001",source)
    (directory/"resume.pt").write_bytes(b"newer")
    remote_cifar.pack(work,data,out,cell,"0002",source)
    restored=tmp_path/"restored"
    assert remote_cifar.restore_archives(out,restored,cell,remote_cifar.sha256(archive))==1
    assert (remote_cifar.run_directory(restored,cell)/"resume.pt").read_bytes()==b"newer"


def test_collector_validates_and_installs_complete_archive(tmp_path,monkeypatch):
    cell=calibration_cell(); work=tmp_path/"work"; data=tmp_path/"data"; cache=tmp_path/"cache"
    repo=tmp_path/"repo"; source=repo/"configs/completion/sources.json"; source.parent.mkdir(parents=True)
    archive=data/"fixture"; archive.parent.mkdir(); archive.write_bytes(b"x")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":archive.name,"bytes":1,"sha256":remote_cifar.sha256(archive)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    config=remote_cifar.config_for(cell)
    atomic_json(directory/"config.json",config)
    atomic_json(directory/"summary.json",{
        "config":config,"run_id":directory.name,"status":"budget_exhausted","history":[]})
    (directory/"final.pt").write_bytes(b"checkpoint")
    (directory/"resume.pt").write_bytes(b"terminal optimizer state")
    remote_cifar.pack(work,data,cache,cell,"job-3",source)
    monkeypatch.setattr(collect_remote_cifar,"download",lambda tag,path:None)
    result=collect_remote_cifar.collect(repo,"test-tag",[cell],cache,source,install=True)
    assert len(result["complete"])==1 and not result["invalid"]
    installed=repo/"results"/"cifar10"/"calibration"/directory.name/"final.pt"
    assert installed.read_bytes()==b"checkpoint"
    assert not (installed.parent/"resume.pt").exists()
    assert collect_remote_cifar.collect(repo,"test-tag",[cell],cache,source,install=True)[
        "installed_files"][remote_cifar.cell_id(cell)]==0
