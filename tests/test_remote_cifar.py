import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from scripts import collect_remote_cifar, remote_cifar, run_cifar_remote_queue
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


def test_plan_transports_production_rate_without_decimal_rounding():
    cell={"stage":"production","dataset":"CIFAR10","gamma":128.,"seed":4,
          "lr":4.525483399593905}
    transported=remote_cifar.plan(json.dumps([cell]))[0]
    assert "lr" not in transported
    assert transported["lr_hex"]==float(cell["lr"]).hex()
    # Simulate the JSON round trip through the GitHub matrix expression engine:
    # the rate remains a string and reconstructs the exact frozen float.
    reconstructed=remote_cifar.parse_cell(json.loads(json.dumps(transported)))
    assert reconstructed==cell
    assert remote_cifar.cell_id(transported)==remote_cifar.cell_id(cell)


def test_production_retry_exclusion_is_exact_and_fail_closed():
    frozen={"selection":{str(gamma):{"lr":.01} for gamma in remote_cifar.GAMMAS}}
    cells=run_cifar_remote_queue.production_cells("CIFAR10",frozen)
    excluded=[remote_cifar.cell_id(cell) for cell in cells if cell["gamma"]==128]
    selected=run_cifar_remote_queue.exclude_production_cells(cells,excluded)
    assert len(selected)==30 and all(cell["gamma"]<128 for cell in selected)
    with pytest.raises(ValueError,match="Unknown excluded"):
        run_cifar_remote_queue.exclude_production_cells(cells,["0"*16])
    with pytest.raises(ValueError,match="Duplicate excluded"):
        run_cifar_remote_queue.exclude_production_cells(cells,[excluded[0],excluded[0]])

    included=run_cifar_remote_queue.include_production_cells(cells,excluded)
    assert len(included)==5 and all(cell["gamma"]==128 for cell in included)
    with pytest.raises(ValueError,match="Unknown included"):
        run_cifar_remote_queue.include_production_cells(cells,["f"*16])
    with pytest.raises(ValueError,match="Duplicate included"):
        run_cifar_remote_queue.include_production_cells(cells,[excluded[0],excluded[0]])


@pytest.mark.parametrize("lr_hex",["not-a-float","nan","inf","-0x1p+0"])
def test_invalid_hex_rate_fails_closed(lr_hex):
    cell={"stage":"production","dataset":"CIFAR10","gamma":128.,"seed":4,
          "lr_hex":lr_hex}
    with pytest.raises(ValueError):
        remote_cifar.parse_cell(cell)


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


def test_production_ph_cache_survives_pack_restore_and_collection(tmp_path,monkeypatch):
    cell={"stage":"production","dataset":"CIFAR10","gamma":.125,
          "seed":0,"lr":.0015625}
    work=tmp_path/"work"; data=tmp_path/"data"; outgoing=tmp_path/"outgoing"
    repo=tmp_path/"repo"; source=repo/"configs/completion/sources.json"
    source.parent.mkdir(parents=True)
    archive=data/"fixture"; archive.parent.mkdir(); archive.write_bytes(b"dataset")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":archive.name,"bytes":len(b"dataset"),
                   "sha256":remote_cifar.sha256(archive)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    config=remote_cifar.config_for(cell)
    atomic_json(directory/"config.json",config)
    atomic_json(directory/"summary.json",{
        "config":config,"run_id":directory.name,"status":"budget_exhausted","history":[]})
    (directory/"final.pt").write_bytes(b"checkpoint")
    (directory/"metrics.json").write_bytes(b"{}")
    (directory/"representations.npz").write_bytes(b"representations")
    cache_member=work/"ph_cache"/("a"*64)/"repeat_000.npz"
    cache_member.parent.mkdir(parents=True); cache_member.write_bytes(b"completed PH repeat")

    report=remote_cifar.pack(work,data,outgoing,cell,"job-ph",source)
    cache_name=cache_member.relative_to(work).as_posix()
    assert report["complete"] and cache_name in report["files"]
    artifact=next(outgoing.glob("cifar-*.tar.gz"))
    resume=tmp_path/"resume"; resume.mkdir(); shutil.copy2(artifact,resume/artifact.name)
    restored=tmp_path/"restored"
    assert remote_cifar.restore_archives(resume,restored,cell,remote_cifar.sha256(archive))==6
    assert (restored/cache_name).read_bytes()==b"completed PH repeat"

    transport=tmp_path/"transport"; transport.mkdir()
    for path in outgoing.iterdir():
        shutil.copy2(path,transport/path.name)
    monkeypatch.setattr(collect_remote_cifar,"download",lambda tag,path:None)
    collection=collect_remote_cifar.collect(repo,"test-tag",[cell],transport,source,install=True)
    assert not collection["invalid"] and len(collection["complete"])==1
    installed=repo/"results"/cache_name
    assert installed.read_bytes()==b"completed PH repeat"


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


def test_bounded_collector_streams_once_and_reuses_validation_ledger(tmp_path,monkeypatch):
    cell=calibration_cell(); work=tmp_path/"work"; data=tmp_path/"data"
    remote=tmp_path/"remote"; cache=tmp_path/"cache"; cache.mkdir()
    repo=tmp_path/"repo"; source=repo/"configs/completion/sources.json"; source.parent.mkdir(parents=True)
    dataset=data/"fixture"; dataset.parent.mkdir(); dataset.write_bytes(b"x")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":dataset.name,"bytes":1,"sha256":remote_cifar.sha256(dataset)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    config=remote_cifar.config_for(cell)
    atomic_json(directory/"config.json",config)
    atomic_json(directory/"summary.json",{
        "config":config,"run_id":directory.name,"status":"budget_exhausted","history":[]})
    (directory/"final.pt").write_bytes(b"checkpoint")
    remote_cifar.pack(work,data,remote,cell,"job-4",source)
    for status in remote.glob("status-*.json"):
        shutil.copy2(status,cache/status.name)
    calls=[]
    monkeypatch.setattr(collect_remote_cifar,"download_statuses",lambda tag,path:None)
    def copy_asset(tag,name,destination):
        calls.append(name); target=Path(destination)/name; shutil.copy2(remote/name,target); return target
    monkeypatch.setattr(collect_remote_cifar,"download_asset",copy_asset)
    result=collect_remote_cifar.collect(repo,"test-tag",[cell],cache,source,
                                        install=True,bounded_cache=True)
    assert len(result["complete"])==1 and not result["invalid"] and len(calls)==1
    assert not list(cache.glob("*.tar.gz"))
    assert (cache/"validated_transport.json").is_file()
    again=collect_remote_cifar.collect(repo,"test-tag",[cell],cache,source,
                                       install=True,bounded_cache=True)
    assert not again["invalid"] and len(calls)==1


def test_missing_partial_archive_is_recorded_and_remains_retryable(tmp_path,monkeypatch):
    cell=calibration_cell(); work=tmp_path/"work"; data=tmp_path/"data"
    remote=tmp_path/"remote"; cache=tmp_path/"cache"; cache.mkdir()
    repo=tmp_path/"repo"; source=repo/"configs/completion/sources.json"; source.parent.mkdir(parents=True)
    dataset=data/"fixture"; dataset.parent.mkdir(); dataset.write_bytes(b"x")
    atomic_json(source,{"schema":"feature-topology.cifar-sources.v1","datasets":{
        "CIFAR10":{"archive":dataset.name,"bytes":1,"sha256":remote_cifar.sha256(dataset)}}})
    directory=remote_cifar.run_directory(work,cell); directory.mkdir(parents=True)
    (directory/"resume.pt").write_bytes(b"partial")
    remote_cifar.pack(work,data,remote,cell,"job-missing",source)
    status=next(remote.glob("status-*.json")); shutil.copy2(status,cache/status.name)
    monkeypatch.setattr(collect_remote_cifar,"download_statuses",lambda tag,path:None)
    def missing(*args,**kwargs):
        raise subprocess.CalledProcessError(1,["gh","release","download"],
                                            stderr="release asset not found")
    monkeypatch.setattr(collect_remote_cifar,"download_asset",missing)
    result=collect_remote_cifar.collect(repo,"test-tag",[cell],cache,source,
                                        install=True,bounded_cache=True)
    identifier=remote_cifar.cell_id(cell)
    assert result["missing"]==[identifier] and not result["invalid"]
    assert result["partial"]=={identifier:[status.name]}
    assert result["unavailable_partial"]==[{
        "status":status.name,
        "archive":next(remote.glob("cifar-*.tar.gz")).name,
        "error":"release asset not found",
    }]


def test_cifar_status_poll_retries_transient_read_only_failure(monkeypatch):
    calls=[]
    def fake_run(command):
        calls.append(command)
        if len(calls)==1:
            raise subprocess.CalledProcessError(1,command)
        return json.dumps({"status":"queued","conclusion":"","url":"https://example/run",
                           "jobs":[{"status":"in_progress"}]})
    monkeypatch.setattr(run_cifar_remote_queue,"run",fake_run)
    monkeypatch.setattr(run_cifar_remote_queue.time,"sleep",lambda _:None)
    state=run_cifar_remote_queue.workflow_state("123",retries=2,initial_delay=0)
    assert len(calls)==2 and state["counts"]=={"in_progress":1}


def test_cifar_status_poll_exhausts_bounded_retries(monkeypatch):
    def fail(command):
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(run_cifar_remote_queue,"run",fail)
    monkeypatch.setattr(run_cifar_remote_queue.time,"sleep",lambda _:None)
    with pytest.raises(subprocess.CalledProcessError):
        run_cifar_remote_queue.workflow_state("123",retries=2,initial_delay=0)


def test_remote_tag_commit_peels_annotated_tags():
    tag="result-tag"
    commit="a"*40
    tag_object="b"*40
    annotated=f"{tag_object}\trefs/tags/{tag}\n{commit}\trefs/tags/{tag}^{{}}"
    lightweight=f"{commit}\trefs/tags/{tag}"
    assert run_cifar_remote_queue.remote_tag_commit(annotated,tag)==commit
    assert run_cifar_remote_queue.remote_tag_commit(lightweight,tag)==commit
