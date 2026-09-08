"""Audit the existing image-runner schema separately from synthetic trajectories.

This is an artifact/coverage audit, not validation of image topology, labels,
rotation symmetry, floating-point training, or the underlying PH implementation.
"""
from __future__ import annotations
from pathlib import Path
import math
from .io import read_json, original_fingerprint, file_digest

IMAGE_GAMMAS = [.125, .5, 1., 4., 16., 64., 128.]


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
                # Existing digits and CIFAR scripts have different native schemas.
                if 'rotation_loops' in metrics:
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
