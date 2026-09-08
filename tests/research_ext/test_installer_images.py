import hashlib
from pathlib import Path
import subprocess
import pytest
from research_ext.install import apply_update
from research_ext.io import atomic_json, original_fingerprint
from research_ext.image_audit import audit_images
from research_ext.workflow import make_plan


def fake_bundle(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','--allow-empty','-qm','test'],check=True)
    base=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'added.py').write_text('answer = 42\n')
    atomic_json(bundle/'BUNDLE_MANIFEST.json',{'schema':'feature-topology.additive-bundle.v1',
               'base_commit':base,'files':{'added.py':hashlib.sha256((bundle/'added.py').read_bytes()).hexdigest()}})
    return repo,bundle


def test_installer_preflight_and_idempotence(tmp_path):
    repo,bundle=fake_bundle(tmp_path)
    assert apply_update(bundle,repo,dry_run=True)['status']=='dry_run'
    assert not (repo/'added.py').exists()
    assert apply_update(bundle,repo)['added_files']==['added.py']
    assert apply_update(bundle,repo)['already_identical']==['added.py']


def test_installer_refuses_overwrite(tmp_path):
    repo,bundle=fake_bundle(tmp_path); (repo/'added.py').write_text('my existing work\n')
    with pytest.raises(ValueError,match='overwritten'): apply_update(bundle,repo)
    assert (repo/'added.py').read_text()=='my existing work\n'


def test_installer_refuses_tamper(tmp_path):
    repo,bundle=fake_bundle(tmp_path); (bundle/'added.py').write_text('tampered\n')
    with pytest.raises(ValueError,match='hash mismatch'): apply_update(bundle,repo)
    assert not (repo/'added.py').exists()


def test_installer_refuses_symlink(tmp_path):
    repo,bundle=fake_bundle(tmp_path); (repo/'added.py').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='symlinked'): apply_update(bundle,repo)


def test_installer_refuses_traversal(tmp_path):
    repo,bundle=fake_bundle(tmp_path)
    atomic_json(bundle/'BUNDLE_MANIFEST.json',{'schema':'feature-topology.additive-bundle.v1',
        'base_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'files':{'../escape.py':'no'}})
    with pytest.raises(ValueError,match='Unsafe'): apply_update(bundle,repo)


def test_empty_image_study_is_not_complete(tmp_path):
    result=audit_images(tmp_path)
    assert not result['artifact_completion'] and result['expected_runs']==35


def test_image_native_artifacts_and_missing_metrics(tmp_path):
    config={'gamma':.5,'seed':0};rid=original_fingerprint(config)
    run=tmp_path/'runs'/rid;run.mkdir(parents=True)
    atomic_json(run/'config.json',config)
    atomic_json(run/'summary.json',{'config':config,'run_id':rid,'status':'converged'})
    # Existence fixture only; this test does not make a research checkpoint.
    (run/'final.pt').write_bytes(b'test-only-checkpoint-existence-fixture')
    result=audit_images(tmp_path,gammas=[.5],seeds=[0]);assert not result['artifact_completion']
    atomic_json(run/'metrics.json',{'rotation_loops':[{}],'probes':{},'test_accuracy':.9})
    assert audit_images(tmp_path,gammas=[.5],seeds=[0])['artifact_completion']


def test_image_plans_no_unrequested_download(tmp_path):
    cifar=make_plan(tmp_path,suite='cifar')
    assert all('--download' not in t['command'] for t in cifar['tasks'])
    digits=make_plan(tmp_path,suite='images',image_device='mps')
    assert 'mps' in digits['tasks'][0]['command']
