"""Metadata fixtures; interval semantics are tested by the separate verifier controls."""
import json

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from flint import arb,ctx
from scripts import collect_fiber_replication as collector
from scripts.recover_release_evidence import digest
from src.metrics.fixed_fiber_verifier import SCOPE as RETENTION_SCOPE
from src.metrics.nuisance_decision_verifier import SCOPE as DECISION_SCOPE
from src.metrics.rational_fixed_fiber_verifier import dyadic_bound
from tests.test_fiber_replication_worker import screen_fixture


def fixture(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    screens=screen_fixture(repo)
    collection=json.loads((screens/'collection.json').read_text())
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=json.loads(index_path.read_text())
    records={row['run_id']:row for row in index['runs']}
    protocol_path=repo/'configs/fiber_replication_v1.json'
    protocol=json.loads(protocol_path.read_text())
    source=repo/'verifier-fixture.py';source.write_text('# metadata-only test control\n')
    monkeypatch.setattr(collector,'source_files',lambda root:[source])
    output=tmp_path/'artifacts';output.mkdir();ctx.prec=128
    for witness in collection['records']:
        rid=witness['run_id'];record=records[rid]
        norm=sum((arb(float.fromhex(v))**2 for line in witness['readout_hex'] for v in line),arb(0)).sqrt()
        factor=dyadic_bound(((1-arb.pi()/4)/norm).lower(),64,False)
        for theta in [1,3,5,7]:
            retained=witness['empirical_screen_passed'] and theta in [1,3]
            positive,negative=theta in [1,7],theta==3
            if not witness['empirical_screen_passed']:
                retention=dict(status='fixed_witness_screen_failed',continuous_certificate=False,
                    theta_fraction=[theta,8],reason='The predeclared fixed numerical readout failed its screen; not evidence of noninjectivity')
            else:
                retention=dict(status='certified_fixed_nuisance_circle' if retained else
                    'fixed_witness_rejected' if theta==5 else 'inconclusive',
                    continuous_certificate=retained,theta_fraction=[theta,8],scope=RETENTION_SCOPE,checked_boxes=1)
                if retained:retention.update(leaf_codes=[''],epsilon_upper_rational={'numerator':str(1<<19),'denominator':str(1<<20)},
                    chord_lower_factor_rational=factor)
            decision=dict(status='certified_correct_class_circle' if positive else
                'target_class_violation' if negative else 'inconclusive',decision_certificate=positive,
                counterexample_certificate=negative,theta_fraction=[theta,8],target_class=theta//2,
                scope=DECISION_SCOPE,checked_boxes=1)
            if positive:
                decision.update(leaf_codes=[''],unbuffered_margin_lower_rational={'numerator':'100','denominator':str(1<<64)},
                    scaled_margin_lower_rational={'numerator':'50','denominator':str(1<<64)},
                    margin_encoding_policy='Prospectively fixed integer-floor half margin at denominator 2^64')
            elif negative:
                decision.update(violation_code='',violating_class=0,
                    point_margin_upper_rational={'numerator':'-1','denominator':str(1<<64)})
            binding=dict(run_id=rid,gamma=witness['gamma'],seed=witness['seed'],selected_step=witness['selected_step'],
                actual_training_loss=witness['actual_training_loss'],theta_fraction=[theta,8],target_class=theta//2,
                checkpoint_sha256=witness['checkpoint_sha256'],input_archive_sha256=record['archive_sha256'],
                input_index_sha256=digest(index_path),analysis_data_sha256=index['analysis_data']['sha256'],
                screen_result_sha256=digest(screens/(rid+'.json')),screen_collection_sha256=digest(screens/'collection.json'),
                protocol_sha256=digest(protocol_path),source_sha256={'verifier-fixture.py':digest(source)})
            joint=retained and positive
            result=dict(schema='feature-topology.fiber-replication-domain.v1',binding=binding,
                retention=retention,decision=decision,joint_retention_and_correct_decision=joint,
                arithmetic={'python_flint':'0.9.0','precision_bits':128},
                host={'source_commit':'b'*40,'workflow_run':'987'},scope=protocol['scope'])
            path=output/(rid+f'-theta-{theta}-8.json');path.write_text(json.dumps(result))
            receipt=dict(schema='feature-topology.fiber-replication-replay.v1',run_id=rid,
                gamma=witness['gamma'],seed=witness['seed'],theta_fraction=[theta,8],result_sha256=digest(path),
                retention_certificate=retained,decision_certificate=positive,counterexample_certificate=negative,
                retention_replay_passed=retained,decision_replay_passed=positive or negative,
                retention_status='passed' if retained else 'not_a_positive_certificate',
                decision_status='passed' if positive or negative else 'not_a_certificate',
                joint_retention_and_correct_decision=joint,
                scope='Separate-process execution of the same Arb verifiers; no independent verifier or Lean proof')
            path.with_name(path.stem+'.replay.json').write_text(json.dumps(receipt))
    return repo,output


def test_all_planned_domains_and_mixed_outcomes_are_preserved(tmp_path,monkeypatch):
    repo,artifacts=fixture(tmp_path,monkeypatch)
    flat=tmp_path/'flat'
    result=collector.collect(repo,artifacts,flat,'987','b'*40)
    assert result['complete_models']==36 and result['complete_domains']==144
    assert result['eligible_fixed_witnesses']==7 and result['failed_fixed_witnesses']==29
    assert result['retention_certificates']==14 and result['correct_decision_certificates']==72
    assert result['decision_counterexamples']==36 and result['inconclusive_decision_domains']==36
    assert result['joint_certificates']==7 and result['all_four_joint_models']==0
    assert result['training_seeds']==list(range(1,10))
    assert collector.collect(repo,flat)==result


@pytest.mark.parametrize('damage,match',[
    ('source','source, witness'),('joint','both positive'),('factor','not justified'),
    ('buffer','fixed positive buffer'),('screen','relabeled'),('receipt','replay receipt')])
def test_altered_claims_are_rejected_even_with_updated_receipt_hash(tmp_path,monkeypatch,damage,match):
    repo,artifacts=fixture(tmp_path,monkeypatch)
    path=next(p for p in artifacts.glob('*-theta-1-8.json')
        if json.loads(p.read_text())['retention']['continuous_certificate'])
    if damage=='screen':
        path=next(p for p in artifacts.glob('*-theta-1-8.json')
            if json.loads(p.read_text())['retention']['status']=='fixed_witness_screen_failed')
    elif damage=='joint':path=path.with_name(path.name.replace('theta-1','theta-3'))
    result=json.loads(path.read_text())
    receipt_path=path.with_name(path.stem+'.replay.json');receipt=json.loads(receipt_path.read_text())
    if damage=='source':result['binding']['checkpoint_sha256']='0'*64
    elif damage=='joint':result['joint_retention_and_correct_decision']=True
    elif damage=='factor':result['retention']['chord_lower_factor_rational']['numerator']=str(10**30)
    elif damage=='buffer':result['decision']['scaled_margin_lower_rational']['numerator']='51'
    elif damage=='screen':result['retention']['continuous_certificate']=True
    else:receipt['counterexample_certificate']=True
    path.write_text(json.dumps(result));receipt['result_sha256']=digest(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match=match):collector.collect(repo,artifacts)


def test_missing_domain_cannot_be_excused_by_a_failed_screen(tmp_path,monkeypatch):
    repo,artifacts=fixture(tmp_path,monkeypatch)
    path=next(p for p in artifacts.glob('*-theta-1-8.json')
        if json.loads(p.read_text())['retention']['status']=='fixed_witness_screen_failed')
    path.unlink()
    with pytest.raises(ValueError,match='incomplete'):
        collector.collect(repo,artifacts)
