"""Extend a passed full original-study freeze with the additional research audit.

This gate cannot replace, skip, or relax the original physical-input gate. It
adds source checks and replays the diagnostic analysis. A negative predictive
finding remains a completed comparison; no novelty or journal readiness follows
from a successful evidence freeze.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.collect_additional_evaluations import validate_cell
from scripts.diagnostic_comparison import comparison_rows, run_comparison
from scripts.freeze_results import verify_manifest
from scripts.recover_release_evidence import digest, load_json, publish
from scripts.remote_nuisance_evaluation import evaluation_plan


def fixed_fiber_paths(repo, original_files):
    """Check the source-bound circle collection and its physical frozen inputs."""
    from scripts.collect_fixed_fiber_results import collect

    index = load_json((repo/'configs/completion/primary_inputs_v3.json').read_bytes())
    arrays = repo/'results/continuous_fiber_inputs/analysis_data.npz'
    if not arrays.is_file() or digest(arrays) != index['analysis_data']['sha256']:
        raise ValueError('Physical continuous-fiber embedding input is missing or changed')
    root = repo/'results/fixed_fiber_verification_v4'
    replayed_collection = collect(repo, root)
    stored = load_json((root/'collection.json').read_bytes())
    if replayed_collection != stored:
        raise ValueError('Fixed-fiber collection differs from the checked artifacts')
    witnesses = load_json((repo/'results/continuous_fiber_feasibility/manifest.json').read_bytes())['runs']
    paths = {arrays}
    for row in witnesses:
        if not row['empirical_screen_passed']:
            continue
        checkpoint = repo/'results/main/runs'/row['run_id']/row['checkpoint']
        relative = checkpoint.relative_to(repo).as_posix()
        if (not checkpoint.resolve().is_relative_to(repo)
                or original_files.get(relative) != row['checkpoint_sha256']
                or digest(checkpoint) != row['checkpoint_sha256']):
            raise ValueError('Fixed-fiber checkpoint is outside the original full freeze')
        for name, expected in row['source_sha256'].items():
            if digest(repo/name) != expected:
                raise ValueError('Fixed-fiber witness-construction source changed: '+name)
    for name in ['continuous_fiber_feasibility', 'continuous_fiber_verification_v1',
                 'fixed_fiber_verification_v2', 'fixed_fiber_verification_v3_attempts',
                 'fixed_fiber_verification_v4']:
        directory = repo/'results'/name
        if not directory.is_dir() or not any(directory.iterdir()):
            raise ValueError('Missing retained continuous-fiber attempt directory: '+name)
        paths.update(path for path in directory.rglob('*') if path.is_file())
    local = repo/'results/fixed_fiber_local_replays_v4'
    expected_names = {row['run_id']+f"-theta-{row['theta_fraction'][0]}-8.json"
                      for row in stored['records']}
    if {path.name for path in local.glob('*.json')} != expected_names:
        raise ValueError('Additional local circle replays are incomplete or duplicated')
    platforms = set()
    for row in stored['records']:
        name = row['run_id']+f"-theta-{row['theta_fraction'][0]}-8.json"
        result = load_json((root/name).read_bytes())
        receipt = load_json((local/name).read_bytes())
        positive = row['continuous_certificate']
        if (receipt.get('schema') != 'feature-topology.fixed-fiber-local-replay.v4'
                or receipt.get('run_id') != row['run_id']
                or receipt.get('theta_fraction') != row['theta_fraction']
                or receipt.get('result_sha256') != row['result_sha256']
                or receipt.get('binding') != result['binding']
                or receipt.get('positive_certificate') is not positive
                or receipt.get('replay_passed') is not positive
                or receipt.get('status') != ('passed' if positive else 'not_a_positive_certificate')
                or receipt.get('arithmetic') != {'python_flint':'0.9.0','precision_bits':128}
                or receipt.get('verifier_sha256') != digest(repo/'scripts/replay_fixed_fiber_collection.py')):
            raise ValueError('Additional local circle replay binding changed')
        platforms.add(receipt['host']['system']+'/'+receipt['host']['machine'])
        paths.add(local/name)
    return paths, dict(protocol_version=4, attempts=16,
                       positive_certificates=stored['positive_certificates'],
                       trained_models=4, training_seeds=[0],
                       source_bound_standalone_replays_checked=True,
                       additional_local_replays=16, local_replay_platforms=sorted(platforms),
                       scope=stored['scope'], replay_scope=stored['replay_scope'])


def decision_paths(repo, original_files):
    """Require every recorded outcome and preserve its positive/negative distinction."""
    from scripts.collect_nuisance_decision_results import collect
    from scripts.replay_nuisance_decision_collection_v2 import derived_claim

    root = repo/'results/nuisance_decision_verification_v1'
    stored = load_json((root/'collection.json').read_bytes())
    if collect(repo, root) != stored:
        raise ValueError('Classifier decision collection differs from checked artifacts')
    local = repo/'results/nuisance_decision_local_replays_v2'
    names = {row['run_id']+f"-theta-{row['theta_fraction'][0]}-8.json" for row in stored['records']}
    claim_names = {name[:-5]+'.local-claim.json' for name in names}
    if {path.name for path in local.glob('*.json')} != names | claim_names:
        raise ValueError('Additional local classifier replays are incomplete or duplicated')
    paths, platforms = set(root.rglob('*.json')), set()
    for row in stored['records']:
        name = row['run_id']+f"-theta-{row['theta_fraction'][0]}-8.json"
        result = load_json((root/name).read_bytes())
        receipt = load_json((local/name).read_bytes())
        claim_path = local/(name[:-5]+'.local-claim.json')
        if load_json(claim_path.read_bytes()) != derived_claim(result, row['result_sha256']):
            raise ValueError('Local classifier claim differs from the fixed half-margin derivation')
        witness = load_json((repo/'results/continuous_fiber_feasibility'/(row['run_id']+'.json')).read_bytes())
        checkpoint = repo/'results/main/runs'/row['run_id']/witness['checkpoint']
        if (not checkpoint.resolve().is_relative_to(repo)
                or original_files.get(checkpoint.relative_to(repo).as_posix()) != result['binding']['checkpoint_sha256']
                or digest(checkpoint) != result['binding']['checkpoint_sha256']):
            raise ValueError('Classifier checkpoint is outside the original full freeze')
        positive, negative = row['decision_certificate'], row['counterexample_certificate']
        certified = positive or negative
        if (receipt.get('schema') != 'feature-topology.nuisance-decision-local-replay.v2'
                or receipt.get('run_id') != row['run_id']
                or receipt.get('theta_fraction') != row['theta_fraction']
                or receipt.get('target_class') != row['target_class']
                or receipt.get('binding') != result['binding']
                or receipt.get('result_sha256') != row['result_sha256']
                or receipt.get('local_claim_sha256') != digest(claim_path)
                or receipt.get('hosted_numerical_margin_replay_claimed') is not False
                or receipt.get('decision_certificate') is not positive
                or receipt.get('counterexample_certificate') is not negative
                or receipt.get('replay_passed') is not certified
                or receipt.get('status') != ('passed' if certified else 'not_a_certificate')
                or receipt.get('arithmetic') != {'python_flint':'0.9.0', 'precision_bits':128}
                or receipt.get('verifier_sha256') != digest(repo/'scripts/replay_nuisance_decision_collection_v2.py')):
            raise ValueError('Additional local classifier replay binding changed')
        platforms.add(receipt['host']['system']+'/'+receipt['host']['machine'])
        paths.add(local/name)
        paths.add(claim_path)
    preflight = repo/'results/nuisance_decision_preflight_v1'
    if not (preflight/'attempt.json').is_file() or not (preflight/'failure_excerpt.log').is_file():
        raise ValueError('Missing retained classifier source-format preflight failure')
    paths.update(path for path in preflight.rglob('*') if path.is_file())
    failed_local = repo/'results/nuisance_decision_local_attempts_v1'
    if not (failed_local/'initial_margin_failure.json').is_file() or not (failed_local/'initial_replayer.py').is_file():
        raise ValueError('Missing retained original Mac numerical-margin replay failure')
    paths.update(path for path in failed_local.rglob('*') if path.is_file())
    paths.update(repo/name for name in ['scripts/collect_nuisance_decision_results.py',
        'scripts/replay_nuisance_decision_collection.py', 'scripts/replay_nuisance_decision_collection_v2.py',
        'configs/nuisance_decision_verifier_v1.json'])
    return paths, dict(protocol_version=1, attempts=stored['complete_attempts'],
        decision_certificates=stored['decision_certificates'],
        counterexample_certificates=stored['counterexample_certificates'],
        inconclusive=stored['inconclusive'],
        joint_retention_and_correct_decision=stored['joint_retention_and_correct_decision'],
        trained_models=4, training_seeds=[0], source_bound_standalone_replays_checked=True,
        additional_local_replays=16, local_replay_platforms=sorted(platforms),
        local_replay_margin_policy='Fixed half-margin derivation, recorded after the initial Mac bound failure',
        hosted_numerical_margin_validated_on_local_platform=False,
        scope=stored['scope'], negative_scope=stored['negative_scope'], replay_scope=stored['replay_scope'])


def replication_paths(repo, original_files):
    """Require all 144 planned outcomes, not only successful new-seed witnesses."""
    from scripts.collect_fiber_replication import collect
    from scripts.validate_fiber_replication_screens import validate

    screens=repo/'results/fiber_replication_screens_v1'
    witnesses=validate(repo,screens)
    root=repo/'results/fiber_replication_verification_v1'
    stored=load_json((root/'collection.json').read_bytes())
    if collect(repo,root)!=stored:
        raise ValueError('Replication collection differs from checked artifacts')
    index=load_json((repo/'configs/completion/primary_inputs_v3.json').read_bytes())
    arrays=repo/'results/continuous_fiber_inputs/analysis_data.npz'
    if not arrays.is_file() or digest(arrays)!=index['analysis_data']['sha256']:
        raise ValueError('Physical replication embedding is missing or changed')
    paths=set(screens.rglob('*.json'))|set(root.rglob('*.json'))|{arrays}
    for row in witnesses['records']:
        checkpoint=repo/'results/main/runs'/row['run_id']/row['checkpoint']
        if (not checkpoint.resolve().is_relative_to(repo)
                or original_files.get(checkpoint.relative_to(repo).as_posix())!=row['checkpoint_sha256']
                or digest(checkpoint)!=row['checkpoint_sha256']):
            raise ValueError('Replication checkpoint is outside the original full freeze')
        paths.add(checkpoint)
    paths.update(repo/name for name in ['scripts/collect_fiber_replication.py',
        'configs/fiber_replication_v1.json','docs/FIBER_REPLICATION_PROTOCOL.md'])
    summary={key:stored[key] for key in ['complete_models','complete_domains','training_seeds',
        'eligible_fixed_witnesses','failed_fixed_witnesses','attempted_retention_domains',
        'retention_certificates','correct_decision_certificates','decision_counterexamples',
        'inconclusive_decision_domains','joint_certificates','all_four_joint_models','by_gamma',
        'source_commit','workflow_run','scope','interpretation']}
    summary['source_bound_standalone_replays_checked']=True
    return paths,summary


def close_replay(actual, expected, path='result'):
    """Compare all replay fields, allowing only cross-platform floating roundoff."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Analysis replay keys differ: '+path)
        for key in actual:
            close_replay(actual[key], expected[key], path+'.'+key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Analysis replay lengths differ: '+path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            close_replay(left, right, f'{path}[{index}]')
    elif (isinstance(actual, (float, int)) and not isinstance(actual, bool)
          and isinstance(expected, (float, int)) and not isinstance(expected, bool)):
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
            raise ValueError('Analysis replay numeric difference: '+path)
    elif isinstance(actual, bool) or isinstance(expected, bool):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('Analysis replay value differs: '+path)
    elif actual != expected:
        raise ValueError('Analysis replay value differs: '+path)


def freeze(repo, original_manifest, evidence, output,
           comparison='results/additional_diagnostic_comparison_portable_v2'):
    repo = Path(repo).resolve()
    original_manifest = Path(original_manifest).resolve()
    comparison_root = Path(comparison)
    if not comparison_root.is_absolute():
        comparison_root = repo/comparison_root
    comparison_root = comparison_root.resolve()
    if not comparison_root.is_relative_to(repo):
        raise ValueError('Additional comparison must be inside the repository')
    original = verify_manifest(repo, original_manifest)
    plan = evaluation_plan(repo)
    protocol_path = repo/'configs/strengthening_diagnostic_protocol_v1.json'
    protocol = load_json(protocol_path.read_bytes())
    paths = set()
    for kind, folder in [('ood', 'ood_evaluation'), ('variability', 'additional_variability')]:
        root = repo/'results'/folder
        collection = load_json((root/'collection.json').read_bytes())
        if (collection.get('kind') != kind or collection.get('expected') != 140
                or collection.get('complete') != 140 or collection.get('missing') != []
                or collection.get('invalid') != []):
            raise ValueError('Additional evaluation collection incomplete: '+kind)
        for row in plan:
            rid = row['record']['run_id']
            result_path, receipt_path = root/(rid+'.json'), root/'receipts'/(rid+'.json')
            receipt = load_json(receipt_path.read_bytes())
            validate_cell(row, load_json(result_path.read_bytes()), receipt, digest(result_path), kind, repo)
            if kind == 'variability':
                for name, expected in receipt['source_sha256'].items():
                    if digest(repo/name) != expected:
                        raise ValueError('Additional metric source changed: '+name)
            elif receipt['evaluator_sha256'] != digest(repo/'scripts/evaluate_nuisance_shift.py'):
                raise ValueError('Nuisance evaluator source changed')
        paths.update(root.rglob('*.json'))
    aliases = repo/'results/checkpoint_aliases'
    rows, hashes, missing = comparison_rows(repo, evidence, repo/'results/ood_evaluation',
                                           repo/'results/additional_variability', aliases, protocol)
    if len(rows) != 140 or missing:
        raise ValueError('Additional comparison has missing cells or undefined predictors')
    result = load_json((comparison_root/'comparison.json').read_bytes())
    stored_rows = load_json((comparison_root/'comparison_rows.json').read_bytes())
    close_replay(rows, stored_rows, 'comparison_rows')
    if (result.get('status') != 'complete_additional_analysis'
            or result.get('protocol_sha256') != digest(protocol_path)
            or result.get('cells') != 140 or result.get('missing') != []
            or result.get('input_sha256') != hashes):
        raise ValueError('Additional comparison is not bound to the complete current source table')
    expected_analysis_sources={name:digest(repo/name) for name in [
        'scripts/diagnostic_comparison.py','research_ext/catalog.py']}
    if result.get('analysis_source_sha256')!=expected_analysis_sources:
        raise ValueError('Additional analysis source provenance differs from frozen code')
    replay = run_comparison(rows, protocol)
    for key, value in replay.items():
        close_replay(value, result[key], key)
    paths.update(aliases.rglob('*.json'))
    paths.update(comparison_root.rglob('*.json'))
    fiber_paths, fiber_summary = fixed_fiber_paths(repo, original['files'])
    paths.update(fiber_paths)
    classifier_paths, classifier_summary = decision_paths(repo, original['files'])
    paths.update(classifier_paths)
    replication_inputs, replication_summary = replication_paths(repo, original['files'])
    paths.update(replication_inputs)
    files = dict(original['files'])
    for path in paths:
        if not path.resolve().is_relative_to(repo):
            raise ValueError('Additional frozen file escapes repository')
        files[str(path.relative_to(repo))] = digest(path)
    for name, expected in hashes.items():
        supplied = Path(name)
        if supplied.is_absolute() or '..' in supplied.parts or files.get(name) != expected:
            raise ValueError('Additional analysis consumed an input outside the combined freeze: '+name)
    combined = dict(original)
    combined.update(files=files, original_freeze_sha256=digest(original_manifest),
                    additional_research=dict(schema='feature-topology.additional-research-freeze.v1',
                        comparison_path=comparison_root.relative_to(repo).as_posix(),
                        protocol_sha256=digest(protocol_path), cells=140,
                        exact_checkpoint_alias_audits=140, analysis_replay_passed=True,
                        incremental_benefit_supported={row['outcome']: row['incremental_benefit_supported']
                                                       for row in result['contrasts']},
                        fixed_continuous_fibers=fiber_summary,
                        nuisance_decisions=classifier_summary,
                        fiber_replication=replication_summary,
                        scope='Complete additional evidence audit; no novelty/acceptance guarantee'))
    output = Path(output)
    if not output.resolve().is_relative_to(repo):
        raise ValueError('Combined freeze must be saved inside the repository')
    publish(output.parent, {output.name: (json.dumps(combined, indent=2, allow_nan=False)+'\n').encode()})
    return dict(ready=True, cells=140, original_files=len(original['files']), combined_files=len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--original-manifest', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparison',default='results/additional_diagnostic_comparison_portable_v2',
                        help='Source-bound comparison directory; use a new path to retain earlier snapshots')
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo,args.original_manifest,args.evidence,args.output,args.comparison)))
