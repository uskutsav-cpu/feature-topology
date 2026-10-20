"""Collect all recorded decision outcomes without converting counterexamples to positives.

This checks artifact identity, source bindings, exact bound encodings and replay
receipts. Numerical proof replay remains a separate execution step.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flint import ctx
from scripts.collect_fixed_fiber_results import collect as collect_retention
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_decision_verifier import source_files
from src.metrics.fixed_fiber_verifier import check_circle_cover
from src.metrics.nuisance_decision_verifier import MARGIN_BITS, SCOPE
from src.metrics.rational_fixed_fiber_verifier import exact_bound


def collect(repo, artifacts, output=None, workflow_run=None, source_commit=None):
    repo, artifacts = Path(repo).resolve(), Path(artifacts)
    root = repo/'results/fixed_fiber_verification_v4'
    retained = collect_retention(repo, root)
    if retained != load_json((root/'collection.json').read_bytes()):
        raise ValueError('Different source-bound retention collection')
    index_path = repo/'configs/completion/primary_inputs_v3.json'
    index = load_json(index_path.read_bytes())
    records = {row['run_id']: row for row in index['runs']}
    protocol_path = repo/'configs/nuisance_decision_verifier_v1.json'
    protocol = load_json(protocol_path.read_bytes())
    sources = {str(path.relative_to(repo)): digest(path) for path in source_files(repo)}
    expected = {(row['run_id'], row['theta_fraction'][0]): row for row in retained['records']}
    if len(expected) != 16 or retained['positive_certificates'] != 16:
        raise ValueError('Expected all sixteen previously certified retention domains')
    seen, files, summary, hosts = set(), {}, [], set()
    ctx.prec = 128
    for path in sorted(artifacts.rglob('*.json')):
        result = load_json(path.read_bytes())
        if 'binding' not in result:
            continue
        binding = result['binding']
        theta = binding['theta_fraction']
        key = (binding['run_id'], theta[0])
        if key not in expected or key in seen or theta != [key[1], 8]:
            raise ValueError('Unexpected or duplicate classifier decision domain')
        seen.add(key)
        rid, numerator = key
        prior, record = expected[key], records[rid]
        witness = load_json((repo/'results/continuous_fiber_feasibility'/(rid+'.json')).read_bytes())
        stem = rid+f'-theta-{numerator}-8'
        retention_path = root/(stem+'.json')
        retention = load_json(retention_path.read_bytes())
        required_binding = dict(run_id=rid, gamma=prior['gamma'], seed=0,
            selected_step=witness['selected_step'], checkpoint_sha256=witness['checkpoint_sha256'],
            input_archive_sha256=record['archive_sha256'], input_index_sha256=digest(index_path),
            analysis_data_sha256=index['analysis_data']['sha256'],
            retention_certificate_sha256=digest(retention_path),
            retention_replay_sha256=digest(root/(stem+'.replay.json')),
            protocol_sha256=digest(protocol_path), theta_fraction=theta,
            target_class=numerator//2, source_sha256=sources)
        if (binding != required_binding
                or binding['checkpoint_sha256'] != retention['binding']['checkpoint_sha256']
                or binding['checkpoint_sha256'] != record['files'][witness['checkpoint']]
                or record['gamma'] != prior['gamma'] or record['seed'] != 0
                or result.get('schema') != 'feature-topology.nuisance-decision-verification.v1'
                or result.get('arithmetic') != {'python_flint':'0.9.0', 'precision_bits':128}
                or result.get('theta_fraction') != theta or result.get('target_class') != numerator//2
                or result.get('scope') != SCOPE):
            raise ValueError('Classifier source, arithmetic, retention or scope binding mismatch')
        host = result.get('host', {})
        commit, run = host.get('source_commit', ''), host.get('workflow_run', '')
        if (not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)
                or not isinstance(run, str) or not re.fullmatch('[1-9][0-9]*', run)
                or (source_commit is not None and commit != source_commit)
                or (workflow_run is not None and run != str(workflow_run))):
            raise ValueError('Different hosted source commit or workflow run')
        hosts.add((commit, run))
        positive, negative = result.get('decision_certificate'), result.get('counterexample_certificate')
        status = result.get('status')
        flags = {'certified_correct_class_circle':(True, False),
                 'target_class_violation':(False, True), 'inconclusive':(False, False)}
        if (status not in flags or type(positive) is not bool or type(negative) is not bool
                or (positive, negative) != flags[status]):
            raise ValueError('Conflicting classifier certificate status or flags')
        checked = result.get('checked_boxes')
        if (type(checked) is not int or not 1 <= checked <= protocol['budget']['maximum_boxes']):
            raise ValueError('Classifier verification box budget differs from the protocol')
        if positive:
            check_circle_cover(result['leaf_codes'])
            if (any(len(code) > protocol['budget']['maximum_depth'] for code in result['leaf_codes'])
                    or not exact_bound(result['scaled_margin_lower_rational'], MARGIN_BITS) > 0):
                raise ValueError('Positive classifier certificate lacks a valid strict margin')
        elif negative:
            code, competitor = result['violation_code'], result['violating_class']
            if (not isinstance(code, str) or any(char not in '01' for char in code)
                    or len(code) > protocol['budget']['maximum_depth']
                    or type(competitor) is not int or competitor not in set(range(4))-{numerator//2}
                    or not exact_bound(result['point_margin_upper_rational'], MARGIN_BITS) < 0):
                raise ValueError('Counterexample lacks a valid negative target-class margin')
        elif any(name in result for name in ['leaf_codes', 'scaled_margin_lower_rational',
                                             'point_margin_upper_rational', 'violation_code']):
            raise ValueError('Inconclusive attempt contains contradictory certificate fields')
        receipt_path = path.with_name(path.stem+'.replay.json')
        receipt = load_json(receipt_path.read_bytes())
        certified = positive or negative
        if (receipt.get('schema') != 'feature-topology.nuisance-decision-replay.v1'
                or receipt.get('run_id') != rid or receipt.get('theta_fraction') != theta
                or receipt.get('target_class') != numerator//2
                or receipt.get('result_sha256') != digest(path)
                or receipt.get('decision_certificate') is not positive
                or receipt.get('counterexample_certificate') is not negative
                or receipt.get('standalone_replay_passed') is not certified
                or receipt.get('status') != ('passed' if certified else 'not_a_certificate')
                or receipt.get('scope') != 'Separate-process execution of the same Arb verifier, not an independent implementation or Lean proof'):
            raise ValueError('Missing or different classifier replay receipt')
        files[stem+'.json'], files[stem+'.replay.json'] = path.read_bytes(), receipt_path.read_bytes()
        summary.append(dict(run_id=rid, gamma=prior['gamma'], seed=0, theta_fraction=theta,
            target_class=numerator//2, status=status, decision_certificate=positive,
            counterexample_certificate=negative, joint_retention_and_correct_decision=positive,
            result_sha256=digest(path), replay_sha256=digest(receipt_path),
            retention_certificate_sha256=binding['retention_certificate_sha256'],
            scaled_margin_lower_rational=result.get('scaled_margin_lower_rational'),
            point_margin_upper_rational=result.get('point_margin_upper_rational')))
    if seen != set(expected) or len(hosts) != 1:
        raise ValueError(f'Classifier decision collection incomplete or mixed sources: {len(seen)}/16')
    summary.sort(key=lambda row:(row['gamma'], row['theta_fraction'][0]))
    commit, run = next(iter(hosts))
    manifest = dict(schema='feature-topology.nuisance-decision-collection.v1',
        expected_attempts=16, complete_attempts=16, trained_models=4, training_seeds=[0],
        decision_certificates=sum(row['decision_certificate'] for row in summary),
        counterexample_certificates=sum(row['counterexample_certificate'] for row in summary),
        inconclusive=sum(row['status']=='inconclusive' for row in summary),
        joint_retention_and_correct_decision=sum(row['joint_retention_and_correct_decision'] for row in summary),
        source_commit=commit, workflow_run=run, protocol_sha256=digest(protocol_path), records=summary,
        scope=SCOPE,
        negative_scope='A negative point refutes universal correct target class on that circle; it does not by itself prove variation in predicted class',
        replay_scope='Every positive or negative claim replayed using the same Arb implementation; not an independent verifier or Lean proof')
    files['collection.json'] = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    if output is not None:
        publish(Path(output), files)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workflow-run', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    result = collect(args.repo, args.artifacts, args.output, args.workflow_run, args.source_commit)
    print(json.dumps({key:result[key] for key in ['complete_attempts', 'decision_certificates',
                                                'counterexample_certificates', 'inconclusive']}))
