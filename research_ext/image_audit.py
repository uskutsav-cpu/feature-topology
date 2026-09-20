"""Audit the existing image-runner schema separately from synthetic trajectories.

This is an artifact/coverage audit, not validation of image topology, labels,
rotation symmetry, floating-point training, or the underlying PH implementation.
"""
from __future__ import annotations
from pathlib import Path
import math
from .io import read_json, original_fingerprint, file_digest

IMAGE_GAMMAS = [.125, .5, 1., 4., 16., 64., 128.]


def nonfinite_paths(value, path=''):
    """Return paths to null or non-finite values in measured JSON trees."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return [path or '<root>']
    if isinstance(value, dict):
        return [item for key, child in value.items()
                for item in nonfinite_paths(child, f'{path}.{key}' if path else key)]
    if isinstance(value, list):
        return [item for index, child in enumerate(value)
                for item in nonfinite_paths(child, f'{path}[{index}]')]
    return []


def explicit_undefined_cka(metrics):
    """Validate declared zero-variance CKA outcomes and return their JSON paths."""
    paths=set()
    allowed={'undefined_zero_variance_initial','undefined_zero_variance_current',
             'undefined_zero_variance_both'}
    for index,layer in enumerate(metrics.get('layers',[])):
        if layer.get('cka_drift') is not None:
            continue
        status=layer.get('cka_status')
        initial=layer.get('cka_initial_centered_gram_norm')
        current=layer.get('cka_current_centered_gram_norm')
        if (status not in allowed or not isinstance(initial,(int,float))
                or isinstance(initial,bool) or not math.isfinite(initial) or initial<0
                or not isinstance(current,(int,float)) or isinstance(current,bool)
                or not math.isfinite(current) or current<0):
            raise ValueError(f'Unexplained undefined CKA at layer {index+1}')
        expected=('undefined_zero_variance_both' if initial==0 and current==0
                  else 'undefined_zero_variance_initial' if initial==0
                  else 'undefined_zero_variance_current' if current==0 else None)
        if status!=expected:
            raise ValueError(f'Inconsistent undefined CKA diagnostics at layer {index+1}')
        paths.add(f'layers[{index}].cka_drift')
    declared=set(metrics.get('explicit_undefined_metrics',[]))
    if declared!=paths:
        raise ValueError('Explicit undefined-metric declaration mismatch')
    return paths


def audit_images(root: str | Path, *, gammas=None, seeds=None) -> dict:
    root = Path(root)
    gammas = list(IMAGE_GAMMAS if gammas is None else gammas)
    seeds = list(range(5) if seeds is None else seeds)
    expected = {(float(g), int(s)) for g in gammas for s in seeds}
    observed, finished, measured, problems, rows, hashes = set(), set(), set(), [], [], {}
    for run in sorted((root/'runs').glob('*')):
        if not run.is_dir() or not (run/'config.json').is_file():
            continue
        try:
            config = read_json(run/'config.json')
            pair = (float(config['gamma']), int(config['seed']))
            if pair in observed:
                raise ValueError('Duplicate gamma/seed pair; separate experiment conditions')
            observed.add(pair)
            rid = original_fingerprint(config)
            row = dict(run_id=rid, gamma=pair[0], seed=pair[1], status='incomplete', metrics_present=False)
            rows.append(row)
            if not (run/'summary.json').is_file():
                continue
            summary = read_json(run/'summary.json')
            if summary['config'] != config or summary['run_id'] != rid:
                raise ValueError('Summary/config identity mismatch')
            invalid_summary = nonfinite_paths(summary)
            if invalid_summary:
                raise ValueError(f'Nonfinite summary values: {invalid_summary}')
            row['status'] = summary['status']
            if row['status'] not in {'converged', 'budget_exhausted', 'diverged'}:
                raise ValueError('Unknown image run status')
            if row['status'] == 'diverged':
                finished.add(pair)
            elif (run/'final.pt').is_file() and (run/'final.pt').stat().st_size:
                finished.add(pair)
            else:
                raise ValueError('Nondiverged image run has no nonempty final checkpoint')
            if (run/'metrics.json').is_file() and row['status'] != 'diverged':
                metrics = read_json(run/'metrics.json')
                explicit=explicit_undefined_cka(metrics)
                invalid_metrics = [path for path in nonfinite_paths(metrics) if path not in explicit]
                if invalid_metrics:
                    raise ValueError(f'Nonfinite metric values: {invalid_metrics}')
                # Existing digits and CIFAR scripts have different native schemas.
                if metrics.get('schema') == 'feature-topology.dsprites-metrics.v1':
                    if len(metrics.get('layers', [])) != 3 or any(
                        set(layer.get('shape_probes', {})) != {'0', '1', '2'} for layer in metrics['layers']):
                        raise ValueError('Incomplete dSprites metrics')
                    if not metrics.get('initial_model_reconstructed_from_seed') or not isinstance(
                        metrics.get('initial_model_sha256'), str):
                        raise ValueError('dSprites metrics lack the seeded initial-model provenance')
                    if metrics.get('dataset_id') != config.get('dataset_id'):
                        raise ValueError('dSprites dataset identity mismatch')
                    if metrics.get('checkpoint_sha256') != file_digest(run/'final.pt'):
                        raise ValueError('dSprites checkpoint hash mismatch')
                    accuracy = metrics['test']['accuracy']
                    row['metric_schema'] = 'dsprites'
                elif 'rotation_loops' in metrics:
                    if not metrics['rotation_loops'] or not isinstance(metrics.get('probes'), dict):
                        raise ValueError('Incomplete rotated-digits metrics')
                    accuracy = metrics['test_accuracy']
                    row['metric_schema'] = 'rotated_digits'
                elif 'layers' in metrics:
                    if len(metrics['layers']) != 4 or not isinstance(metrics.get('test'), dict):
                        raise ValueError('Incomplete CIFAR metrics')
                    accuracy = metrics['test'].get('accuracy')
                    row['metric_schema'] = 'cifar'
                else:
                    raise ValueError('Unknown image metric schema')
                if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                    raise ValueError('Invalid test accuracy')
                row.update(metrics_present=True, test_accuracy=accuracy)
                row['explicit_undefined_metrics']=sorted(explicit)
                measured.add(pair)
            for filename in ['config.json', 'summary.json', 'metrics.json']:
                path = run/filename
                if path.is_file():
                    hashes[str(path)] = file_digest(path)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            problems.append({'run':str(run), 'error':str(exc)})
    divergent = {(r['gamma'], r['seed']) for r in rows if r['status'] == 'diverged'}
    complete = not problems and finished == expected and (measured | divergent) == expected and observed == expected
    return {'schema':'feature-topology.image-audit.v1', 'artifact_completion':complete,
            'expected_runs':len(expected), 'observed_runs':len(observed), 'finished_runs':len(finished & expected),
            'measured_runs':len(measured & expected), 'diverged_runs':len(divergent & expected),
            'missing_pairs':[list(x) for x in sorted(expected-observed)],
            'unexpected_pairs':[list(x) for x in sorted(observed-expected)],
            'missing_metrics_pairs':[list(x) for x in sorted(expected-measured-divergent)],
            'issues':problems, 'runs':rows, 'input_hashes':hashes,
            'scope':'Native image artifact coverage only; no injectivity, continuum topology, or research-completion claim.'}
