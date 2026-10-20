"""Summarize additional research only after its combined physical-input freeze."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.final_analysis import verify_frozen_paths
from scripts.freeze_results import verify_manifest
from scripts.recover_release_evidence import digest, load_json
from src.training.checkpoints import atomic_json, fingerprint


def metric_row(record, witness, summary, metrics):
    eligible = [row for row in summary['history'] if row['training_loss'] <= .1]
    selected = min(eligible, key=lambda row: row['step']) if eligible else None
    if (selected is None or selected['step'] != witness['selected_step']
            or metrics.get('step') != selected['step']
            or metrics.get('checkpoint_sha256') != witness['checkpoint_sha256']
            or metrics.get('gamma') != record['gamma'] or metrics.get('seed') != 0):
        raise ValueError('Continuous-fiber metric/checkpoint selection mismatch')
    layers = [row for row in metrics['layers'] if row['layer'] == record['config']['depth']]
    if len(layers) != 1:
        raise ValueError('Continuous-fiber metric has an ambiguous last hidden layer')
    layer = layers[0]
    values = dict(centered_rms_scale=layer['scale'], effective_rank=layer['effective_rank'],
        sampled_fiber_margin=layer['fiber_global_separation']['sampled_minimum'],
        sampled_normalized_fiber_margin=layer['fiber_global_separation']['normalized_sampled_minimum'],
        held_out_linear_nuisance_cosine=layer['probes']['linear']['angular_cosine'])
    if not all(np.isfinite(value) for value in values.values()) or values['centered_rms_scale'] <= 0:
        raise ValueError('Continuous-fiber model summary contains invalid sampled metrics')
    return dict(run_id=record['run_id'], gamma=record['gamma'], seed=0,
        selected_step=selected['step'], achieved_training_loss=selected['training_loss'],
        empirical_screen_passed=witness['empirical_screen_passed'],
        sampled_metric_scope='Original frozen grid/probe calculations; these values are not continuum bounds',
        **values)


def summarize(repo, manifest):
    additional = manifest.get('additional_research', {})
    fiber_audit = additional.get('fixed_continuous_fibers', {})
    if (not manifest.get('ready') or fiber_audit.get('attempts') != 16
            or fiber_audit.get('source_bound_standalone_replays_checked') is not True
            or additional.get('analysis_replay_passed') is not True):
        raise ValueError('A complete combined additional-research freeze is required')
    index_path = repo/'configs/completion/primary_inputs_v3.json'
    witness_root = repo/'results/continuous_fiber_feasibility'
    circle_root = repo/'results/fixed_fiber_verification_v4'
    comparison_path = repo/additional['comparison_path']/'comparison.json'
    consumed = verify_frozen_paths(repo, manifest, [index_path, witness_root/'manifest.json',
        circle_root/'collection.json', comparison_path, repo/'scripts/analyze_additional_research.py'])
    index = load_json(index_path.read_bytes())
    witnesses = load_json((witness_root/'manifest.json').read_bytes())['runs']
    collection = load_json((circle_root/'collection.json').read_bytes())
    primary = {row['run_id']: row for row in index['runs'] if row['seed'] == 0}
    if (len(witnesses) != 13 or {row['run_id'] for row in witnesses} != set(primary)
            or collection.get('complete_attempts') != 16):
        raise ValueError('Incomplete primary screening or fixed-circle collection')
    rows = []
    for witness in sorted(witnesses, key=lambda row: row['gamma']):
        rid = witness['run_id']
        record = primary[rid]
        run = repo/'results/main/runs'/rid
        metric_root = run/'metrics'/fingerprint(index['metric_options'])
        metric = metric_root/(Path(witness['checkpoint']).stem+'.json')
        consumed.update(verify_frozen_paths(repo, manifest, [witness_root/(rid+'.json'),
            run/'summary.json', run/'config.json', run/witness['checkpoint'],
            metric_root/'options.json', metric]))
        if load_json((metric_root/'options.json').read_bytes()) != index['metric_options']:
            raise ValueError('Different production metric options')
        if (load_json((run/'config.json').read_bytes()) != record['config']
                or digest(run/witness['checkpoint']) != witness['checkpoint_sha256']):
            raise ValueError('Different source model for the continuous-fiber summary')
        row = metric_row(record, witness, load_json((run/'summary.json').read_bytes()),
                         load_json(metric.read_bytes()))
        certificates = []
        for attempt in collection['records']:
            if attempt['run_id'] != rid:
                continue
            stem = rid+f"-theta-{attempt['theta_fraction'][0]}-8"
            consumed.update(verify_frozen_paths(repo, manifest, [circle_root/(stem+'.json'),
                                                                circle_root/(stem+'.replay.json')]))
            certificates.append({key: attempt[key] for key in [
                'theta_fraction', 'status', 'continuous_certificate',
                'epsilon_upper_rational', 'chord_lower_factor_rational']})
        row['specified_circle_attempts'] = certificates
        rows.append(row)
    comparison = load_json(comparison_path.read_bytes())
    return dict(schema='feature-topology.additional-research-summary.v1', models=rows,
        predictive_contrasts=comparison['contrasts'], independent_prediction_seeds=5,
        screening_models=13, eligible_models=sum(row['empirical_screen_passed'] for row in rows),
        circle_models=4, circle_training_seeds=[0], specified_circles=16,
        positive_circle_certificates=collection['positive_certificates'], input_sha256=consumed,
        scope='Descriptive exploratory seed-zero circle evidence and a separately specified five-seed predictive comparison; no universal-theta, full-torus, independent-verifier or high-novelty claim')


def fraction_float(encoded):
    return float(Fraction(int(encoded['numerator']), int(encoded['denominator'])))


def figures(result, output):
    rows = result['models']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot([row['gamma'] for row in rows],
                 [row['sampled_normalized_fiber_margin'] for row in rows], 'o-', color='#285a83')
    eligible = [row for row in rows if row['empirical_screen_passed']]
    axes[0].scatter([row['gamma'] for row in eligible],
        [row['sampled_normalized_fiber_margin'] for row in eligible], s=90, facecolors='none',
        edgecolors='#bd4f24', linewidths=1.8, label='Eligible fixed readout (4/13)')
    axes[0].set(xscale='log', yscale='log', xlabel='Output scaling gamma',
        ylabel='Sampled fiber margin / recorded RMS', title='A  Grid evidence, seed 0')
    axes[0].legend(loc='lower left', fontsize=8)
    colors = ['#285a83', '#bd4f24', '#39846b', '#8060a3']
    for model_index, row in enumerate(eligible):
        for attempt in row['specified_circle_attempts']:
            if attempt['continuous_certificate']:
                axes[1].scatter(attempt['theta_fraction'][0],
                    fraction_float(attempt['chord_lower_factor_rational']),
                    marker=['o','s','^','D'][model_index], s=55, color=colors[model_index],
                    label=f"gamma {row['gamma']:g}" if attempt['theta_fraction'][0] == 1 else None)
    axes[1].set(yscale='log', xticks=[1,3,5,7], xticklabels=['pi/4','3pi/4','5pi/4','7pi/4'],
        xlabel='Specified fixed task coordinate theta', ylabel='Certified raw chord separation lower factor',
        title='B  Whole-circle bounds, four saved models')
    axes[1].legend(fontsize=8)
    for suffix in ['png', 'svg']:
        fig.savefig(output/f'continuous_fiber_audit.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for ax, row, label in zip(axes, result['predictive_contrasts'],
                              ['Concentrated shift', 'Spurious shift', 'Unseen arc']):
        point = row['conventional_minus_fiber_mse']
        lo, hi = row['family_adjusted_interval']
        ax.errorbar([0], [point], yerr=[[point-lo], [hi-point]], fmt='o', capsize=5, color='#285a83')
        ax.scatter(np.linspace(-.12,.12,5), list(row['seed_differences'].values()),
                   color='#666666', s=20, alpha=.8)
        ax.axhline(0, color='#777777', linewidth=1, linestyle='--')
        ax.set(title=label, xticks=[], xlim=(-.25,.25))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[0].set_ylabel('Conventional MSE minus fiber-augmented MSE')
    fig.supxlabel('Negative values favor conventional predictors; dots are five held-out seeds.\nIntervals: seed bootstrap with Bonferroni adjustment for three outcomes.', fontsize=9)
    for suffix in ['png', 'svg']:
        fig.savefig(output/f'additional_prediction_comparison.{suffix}', dpi=180)
    plt.close(fig)


def main(repo, manifest_path, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    manifest = verify_manifest(repo, manifest_path)
    result = summarize(repo, manifest)
    if not output.is_relative_to(repo) or (output.exists() and any(output.iterdir())):
        raise ValueError('Additional analysis needs a new empty directory inside the repository')
    output.mkdir(parents=True, exist_ok=True)
    figures(result, output)
    result['combined_freeze_sha256'] = digest(manifest_path)
    result['figure_sha256'] = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    verify_manifest(repo, manifest_path)
    atomic_json(output/'analysis.json', result)
    return {key: result[key] for key in ['screening_models', 'eligible_models', 'positive_circle_certificates']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.repo, args.manifest, args.output)))
