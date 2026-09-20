"""Separate seed summaries for the distinct image-study metric schemas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.statistics import bootstrap_mean
from src.training.checkpoints import atomic_json


def image_rows(repo):
    rows=[]; exclusions=[]
    for study in ['rotated_digits','dsprites','cifar10','cifar100']:
        for path in sorted((Path(repo)/'results'/study).glob('runs/*/summary.json')):
            summary=json.loads(path.read_text())
            config=summary['config']
            if summary['status']=='diverged':
                continue
            metrics=json.loads((path.parent/'metrics.json').read_text())
            base=dict(study=study,run_id=path.parent.name,gamma=config['gamma'],seed=config['seed'],
                      status=summary['status'],training_loss=summary['history'][-1]['training_loss'])
            values=[]
            if study=='rotated_digits':
                values=[('test_accuracy',0,metrics['test_accuracy']),('test_loss',0,metrics['test_loss']),
                        ('cka_drift',3,metrics['cka_drift']),('effective_rank',3,metrics['effective_rank'])]
                for probe in ['linear','mlp']:
                    values.append((probe+'_orientation_cosine',3,metrics['probes'][probe]['angular_cosine']))
                values.append(('loop_h1_lifetime',3,float(np.mean([v['final_ph']['H1']['top1'] for v in metrics['rotation_loops']]))))
            else:
                values=[('test_accuracy',0,metrics['test']['accuracy']),('test_loss',0,metrics['test']['loss'])]
                for layer in metrics['layers']:
                    if layer['cka_drift'] is None:
                        exclusions.append(dict(study=study,run_id=path.parent.name,
                            gamma=config['gamma'],seed=config['seed'],metric='cka_drift',
                            layer=layer['layer'],reason=layer.get('cka_status','undeclared')))
                    else:
                        values.append(('cka_drift',layer['layer'],layer['cka_drift']))
                    values.append(('effective_rank',layer['layer'],layer['effective_rank']))
                    values.append(('ph_h1_lifetime',layer['layer'],float(np.mean([v['H1']['top1'] for v in layer['persistence']]))))
                    if study=='dsprites':
                        for shape,probes in layer['shape_probes'].items():
                            for probe in ['linear','mlp']:
                                values.append((f'{probe}_shape{shape}_orientation_cosine',layer['layer'],
                                               probes['nuisance_probes'][probe]['angular_cosine']))
            rows.extend({**base,'metric':metric,'layer':layer,'value':value} for metric,layer,value in values)
    return pd.DataFrame(rows),pd.DataFrame(exclusions)


def summarize_images(repo,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame,exclusions=image_rows(repo)
    frame.to_csv(output/'image_metrics_long.csv',index=False)
    exclusions.to_csv(output/'image_metric_exclusions.csv',index=False)
    summaries=[]
    contrasts=[]
    for (study,metric,layer),part in frame.groupby(['study','metric','layer']):
        for gamma,group in part.groupby('gamma'):
            if group['seed'].duplicated().any() or not np.isfinite(group['value']).all():
                raise ValueError('Duplicate seeds or nonfinite image metrics')
            summaries.append(dict(study=study,metric=metric,layer=int(layer),gamma=float(gamma),
                                  **bootstrap_mean(group['value'],repeats=2000)))
        matrix=part.pivot(index='seed',columns='gamma',values='value')
        reference=min(matrix.columns)
        for gamma in matrix.columns:
            paired=matrix[[reference,gamma]].dropna() if gamma!=reference else matrix[[reference]].dropna()
            delta=paired[gamma]-paired[reference]
            contrasts.append(dict(study=study,metric=metric,layer=int(layer),gamma=float(gamma),
                                  reference_gamma=float(reference),**bootstrap_mean(delta,repeats=2000)))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/'image_seed_confidence_intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'image_paired_contrasts.csv',index=False)
    for study,part in summary.groupby('study'):
        metrics=['test_accuracy','cka_drift','effective_rank']
        fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
        for ax,metric in zip(axes,metrics):
            for layer,group in part[part.metric==metric].groupby('layer'):
                group=group.sort_values('gamma')
                ax.plot(group.gamma,group['mean'],marker='o',label='Output' if layer==0 else f'Layer {layer}')
                ax.fill_between(group.gamma,group.lower,group.upper,alpha=.16)
            ax.set_xscale('log',base=2)
            ax.set_xlabel('Output scale γ')
            ax.set_title(metric.replace('_',' ').capitalize())
            ax.grid(alpha=.2)
        axes[-1].legend(frameon=False,fontsize=8)
        fig.suptitle(study+' — mean and pointwise 95% seed bootstrap intervals')
        for suffix in ['png','pdf','svg']:
            fig.savefig(output/f'{study}_overview.{suffix}',dpi=220)
        plt.close(fig)
    atomic_json(output/'image_analysis_scope.json',dict(
        replicate_unit='Training seed; paired contrasts preserve seed pairing across gamma',
        intervals='Pointwise percentile bootstrap, 2000 resamples; not simultaneous intervals',
        risk='Final stopping-threshold comparison; actual training losses retained, not identical-risk matching',
        nonconvergence='All finite nondiverged runs retained with status; divergent outcomes remain in frozen run summaries',
        undefined_metrics='Mathematically undefined CKA from an exactly zero-variance representation is retained as a structural outcome, excluded from numeric CKA summaries only, and listed in image_metric_exclusions.csv; per-cell n is reported.',
        limitations='Image schemas remain separate. No latent-manifold topology claim for CIFAR. dSprites orientation uses shape-specific symmetry harmonics.'))
