"""Account for every replication domain, including failed screens and negative claims."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flint import arb,ctx
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.remote_fiber_replication import source_files
from scripts.screen_fiber_replication import replication_records
from scripts.validate_fiber_replication_screens import validate
from src.metrics.fixed_fiber_verifier import SCOPE as RETENTION_SCOPE,check_circle_cover
from src.metrics.nuisance_decision_verifier import SCOPE as DECISION_SCOPE
from src.metrics.rational_fixed_fiber_verifier import exact_bound


def check_retention(result,witness,theta,protocol):
    positive=result.get('continuous_certificate')
    if type(positive) is not bool or result.get('theta_fraction')!=[theta,8]:
        raise ValueError('Replication retention flags or domain changed')
    if not witness['empirical_screen_passed']:
        if result!=dict(status='fixed_witness_screen_failed',continuous_certificate=False,
            theta_fraction=[theta,8],reason='The predeclared fixed numerical readout failed its screen; not evidence of noninjectivity'):
            raise ValueError('Failed replication screen was relabeled as a proof attempt')
        return False
    if result.get('scope')!=RETENTION_SCOPE:
        raise ValueError('Replication retention scope changed')
    if (type(result.get('checked_boxes')) is not int
            or not 1<=result['checked_boxes']<=protocol['budget']['maximum_boxes']):
        raise ValueError('Replication retention budget changed')
    if positive:
        if result.get('status')!='certified_fixed_nuisance_circle':
            raise ValueError('Replication retention status contradicts its certificate')
        check_circle_cover(result['leaf_codes'])
        if any(len(code)>protocol['budget']['maximum_depth'] for code in result['leaf_codes']):
            raise ValueError('Replication retention cover exceeds the fixed depth')
        epsilon=exact_bound(result['epsilon_upper_rational'],20)
        factor=exact_bound(result['chord_lower_factor_rational'],64)
        norm=sum((arb(float.fromhex(value))**2 for row in witness['readout_hex'] for value in row),arb(0)).sqrt()
        if (not 0<epsilon<2/arb.pi() or not factor>0
                or not factor<=(1-epsilon*arb.pi()/2)/norm):
            raise ValueError('Replication retention separation bound is not justified')
    elif result.get('status') not in {'fixed_witness_rejected','inconclusive'}:
        raise ValueError('Unknown replication retention attempt status')
    elif any(name in result for name in ['leaf_codes','epsilon_upper_rational','chord_lower_factor_rational']):
        raise ValueError('Uncertified replication retention contains contradictory proof fields')
    return positive


def check_decision(result,theta,protocol):
    positive,negative=result.get('decision_certificate'),result.get('counterexample_certificate')
    statuses={'certified_correct_class_circle':(True,False),'target_class_violation':(False,True),
              'inconclusive':(False,False)}
    if (type(positive) is not bool or type(negative) is not bool
            or result.get('status') not in statuses
            or (positive,negative)!=statuses[result['status']]
            or result.get('theta_fraction')!=[theta,8] or result.get('target_class')!=theta//2
            or result.get('scope')!=DECISION_SCOPE):
        raise ValueError('Replication decision flags, status, domain or scope changed')
    checked=result.get('checked_boxes')
    if type(checked) is not int or not 1<=checked<=protocol['budget']['maximum_boxes']:
        raise ValueError('Replication decision budget changed')
    if positive:
        check_circle_cover(result['leaf_codes'])
        original=result['unbuffered_margin_lower_rational']
        exact_bound(original,64)
        expected=dict(numerator=str(int(original['numerator'])//2),denominator=str(1<<64))
        if (result['scaled_margin_lower_rational']!=expected
                or not exact_bound(expected,64)>0
                or any(len(code)>protocol['budget']['maximum_depth'] for code in result['leaf_codes'])
                or result.get('margin_encoding_policy')!='Prospectively fixed integer-floor half margin at denominator 2^64'):
            raise ValueError('Replication decision margin differs from the fixed positive buffer')
    elif negative:
        competitor,code=result['violating_class'],result['violation_code']
        if (type(competitor) is not int or competitor not in set(range(4))-{theta//2}
                or not isinstance(code,str) or any(c not in '01' for c in code)
                or len(code)>protocol['budget']['maximum_depth']
                or not exact_bound(result['point_margin_upper_rational'],64)<0):
            raise ValueError('Replication decision counterexample has no valid negative point bound')
    elif any(key in result for key in ['leaf_codes','scaled_margin_lower_rational','point_margin_upper_rational']):
        raise ValueError('Inconclusive replication decision contains contradictory proof fields')
    return positive,negative


def collect(repo,artifacts,output=None,workflow_run=None,source_commit=None):
    repo,artifacts=Path(repo).resolve(),Path(artifacts)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    protocol_path=repo/'configs/fiber_replication_v1.json'
    protocol=load_json(protocol_path.read_bytes())
    planned=replication_records(index,protocol)
    records={row['run_id']:row for row in planned}
    screens_root=repo/'results/fiber_replication_screens_v1'
    screens=validate(repo,screens_root)
    witnesses={row['run_id']:row for row in screens['records']}
    expected={(rid,theta) for rid in records for theta in protocol['theta_numerators']}
    sources={path.relative_to(repo).as_posix():digest(path) for path in source_files(repo)}
    seen,hosts,files,rows=set(),set(),{},[]
    ctx.prec=128
    for path in sorted(artifacts.rglob('*.json')):
        result=load_json(path.read_bytes())
        if 'binding' not in result:continue
        binding=result['binding'];rid=binding['run_id'];theta=binding['theta_fraction'][0]
        key=(rid,theta)
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate replication domain')
        seen.add(key);record,witness=records[rid],witnesses[rid]
        required=dict(run_id=rid,gamma=record['gamma'],seed=record['seed'],
            selected_step=witness['selected_step'],actual_training_loss=witness['actual_training_loss'],
            theta_fraction=[theta,8],target_class=theta//2,
            checkpoint_sha256=witness['checkpoint_sha256'],input_archive_sha256=record['archive_sha256'],
            input_index_sha256=digest(index_path),analysis_data_sha256=index['analysis_data']['sha256'],
            screen_result_sha256=digest(screens_root/(rid+'.json')),
            screen_collection_sha256=digest(screens_root/'collection.json'),protocol_sha256=digest(protocol_path),
            source_sha256=sources)
        if (result.get('schema')!='feature-topology.fiber-replication-domain.v1' or binding!=required
                or result.get('arithmetic')!={'python_flint':'0.9.0','precision_bits':128}
                or result.get('scope')!=protocol['scope']):
            raise ValueError('Replication source, witness, arithmetic or scope binding mismatch')
        host=result.get('host',{});commit,run=host.get('source_commit',''),host.get('workflow_run','')
        if (not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit)
                or not isinstance(run,str) or not re.fullmatch('[1-9][0-9]*',run)
                or (source_commit is not None and source_commit!=commit)
                or (workflow_run is not None and str(workflow_run)!=run)):
            raise ValueError('Replication hosted source commit or workflow mismatch')
        hosts.add((commit,run))
        retained=check_retention(result['retention'],witness,theta,protocol)
        positive,negative=check_decision(result['decision'],theta,protocol)
        joint=retained and positive
        if result.get('joint_retention_and_correct_decision') is not joint:
            raise ValueError('Replication joint claim requires both positive properties')
        receipt_path=path.with_name(path.stem+'.replay.json')
        receipt=load_json(receipt_path.read_bytes())
        if (receipt.get('schema')!='feature-topology.fiber-replication-replay.v1'
                or receipt.get('run_id')!=rid or receipt.get('gamma')!=record['gamma']
                or receipt.get('seed')!=record['seed'] or receipt.get('theta_fraction')!=[theta,8]
                or receipt.get('result_sha256')!=digest(path)
                or receipt.get('retention_certificate') is not retained
                or receipt.get('decision_certificate') is not positive
                or receipt.get('counterexample_certificate') is not negative
                or receipt.get('retention_replay_passed') is not retained
                or receipt.get('decision_replay_passed') is not (positive or negative)
                or receipt.get('retention_status')!=('passed' if retained else 'not_a_positive_certificate')
                or receipt.get('decision_status')!=('passed' if positive or negative else 'not_a_certificate')
                or receipt.get('joint_retention_and_correct_decision') is not joint
                or receipt.get('scope')!='Separate-process execution of the same Arb verifiers; no independent verifier or Lean proof'):
            raise ValueError('Missing or different replication replay receipt')
        stem=rid+f'-theta-{theta}-8'
        files[stem+'.json'],files[stem+'.replay.json']=path.read_bytes(),receipt_path.read_bytes()
        rows.append(dict(run_id=rid,gamma=record['gamma'],seed=record['seed'],theta_fraction=[theta,8],
            screen_passed=witness['empirical_screen_passed'],retention_status=result['retention']['status'],
            decision_status=result['decision']['status'],retention_certificate=retained,
            decision_certificate=positive,counterexample_certificate=negative,
            joint_retention_and_correct_decision=joint,result_sha256=digest(path),replay_sha256=digest(receipt_path)))
    if seen!=expected or len(hosts)!=1:
        raise ValueError(f'Replication collection incomplete or mixed hosted sources: {len(seen)}/144')
    rows.sort(key=lambda row:(row['gamma'],row['seed'],row['theta_fraction'][0]))
    models=[]
    for record in planned:
        part=[row for row in rows if row['run_id']==record['run_id']]
        models.append(dict(run_id=record['run_id'],gamma=record['gamma'],seed=record['seed'],
            fixed_witness_screen_passed=witnesses[record['run_id']]['empirical_screen_passed'],
            specified_domains=4,retention_certificates=sum(row['retention_certificate'] for row in part),
            correct_decision_certificates=sum(row['decision_certificate'] for row in part),
            decision_counterexamples=sum(row['counterexample_certificate'] for row in part),
            joint_certificates=sum(row['joint_retention_and_correct_decision'] for row in part),
            all_four_joint=all(row['joint_retention_and_correct_decision'] for row in part)))
    by_gamma=[]
    for gamma in protocol['gammas']:
        group=[row for row in models if row['gamma']==gamma]
        by_gamma.append(dict(gamma=gamma,training_seeds=9,screen_passed_models=sum(row['fixed_witness_screen_passed'] for row in group),
            all_four_joint_models=sum(row['all_four_joint'] for row in group),
            joint_certified_domains=sum(row['joint_certificates'] for row in group),planned_domains=36,
            correct_decision_certificates=sum(row['correct_decision_certificates'] for row in group),
            decision_counterexamples=sum(row['decision_counterexamples'] for row in group)))
    commit,run=next(iter(hosts))
    manifest=dict(schema='feature-topology.fiber-replication-collection.v1',expected_models=36,complete_models=36,
        expected_domains=144,complete_domains=144,training_seeds=list(range(1,10)),
        eligible_fixed_witnesses=screens['eligible_models'],failed_fixed_witnesses=screens['screen_failed_models'],
        attempted_retention_domains=4*screens['eligible_models'],
        retention_certificates=sum(row['retention_certificate'] for row in rows),
        correct_decision_certificates=sum(row['decision_certificate'] for row in rows),
        decision_counterexamples=sum(row['counterexample_certificate'] for row in rows),
        inconclusive_decision_domains=sum(row['decision_status']=='inconclusive' for row in rows),
        joint_certificates=sum(row['joint_retention_and_correct_decision'] for row in rows),
        all_four_joint_models=sum(row['all_four_joint'] for row in models),
        source_commit=commit,workflow_run=run,protocol_sha256=digest(protocol_path),
        records=rows,models=models,by_gamma=by_gamma,scope=protocol['scope'],
        interpretation='Verification counts describe this fixed cohort and witness procedure, not prevalence of true injectivity. Failed screens and inconclusive proofs are not noninjectivity certificates; specified circles do not increase the nine-seed replication count.')
    files['collection.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    if output is not None:publish(Path(output),files)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--artifacts',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workflow-run',required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    result=collect(args.repo,args.artifacts,args.output,args.workflow_run,args.source_commit)
    print(json.dumps({key:result[key] for key in ['complete_domains','retention_certificates',
        'correct_decision_certificates','decision_counterexamples','joint_certificates','all_four_joint_models']}))
