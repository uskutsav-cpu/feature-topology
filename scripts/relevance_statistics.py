"""Frozen cross-relevance summaries from condition-level matched-risk tables."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research_ext.io import atomic_json, atomic_text
from research_ext.report import METRICS
from research_ext.statistics import bootstrap_mean, paired_contrast


def summarize_relevance(groups, output, repeats=2000):
    """Compare predeclared relevance levels while preserving training-seed pairs.

    ``groups`` is an iterable of ``(relevance, analysis_directory)`` pairs.  Each
    directory must contain the matched-risk table produced by the frozen generic
    analysis.  No thresholds, model selection, or outcome-dependent exclusions
    are introduced here.
    """
    frames=[]
    seen=set()
    for relevance,directory in groups:
        relevance=float(relevance)
        if relevance in seen:
            raise ValueError(f'Duplicate relevance level: {relevance}')
        seen.add(relevance)
        path=Path(directory)/'matched_risk.csv'
        frame=pd.read_csv(path)
        frame['relevance']=relevance
        frames.append(frame)
    if not frames or 0. not in seen:
        raise ValueError('Relevance analysis requires the pure-nuisance baseline')
    frame=pd.concat(frames,ignore_index=True)
    required={'run_id','gamma','seed','layer','relevance',*METRICS}
    if not required<=set(frame):
        raise ValueError(f'Matched-risk relevance columns missing: {sorted(required-set(frame))}')
    keys=['relevance','gamma','seed','layer']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate relevance/gamma/seed/layer matched-risk rows')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    atomic_text(output/'relevance_matched_risk_long.csv',frame.to_csv(index=False))
    summaries=[]
    contrasts=[]
    levels=sorted(seen)
    pairs=[('adjacent',left,right) for left,right in zip(levels,levels[1:])]
    pairs += [('baseline',0.,right) for right in levels if right!=0.]
    for (gamma,layer),part in frame.groupby(['gamma','layer']):
        for metric in METRICS:
            for relevance,group in part.groupby('relevance'):
                values=group[metric].dropna()
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite relevance metric: {metric}')
                summaries.append(dict(gamma=float(gamma),layer=int(layer),metric=metric,
                                      relevance=float(relevance),missing_seeds=len(group)-len(values),
                                      **bootstrap_mean(values.tolist(),repeats=repeats)))
            for comparison,left,right in pairs:
                a=part[part.relevance==left].set_index('seed')[metric].dropna().to_dict()
                b=part[part.relevance==right].set_index('seed')[metric].dropna().to_dict()
                contrasts.append(dict(gamma=float(gamma),layer=int(layer),metric=metric,
                                      comparison=comparison,relevance_left=left,relevance_right=right,
                                      **paired_contrast(a,b,repeats=repeats)))
    atomic_text(output/'relevance_seed_confidence_intervals.csv',pd.DataFrame(summaries).to_csv(index=False))
    atomic_json(output/'relevance_paired_contrasts.json',contrasts)
    summary=pd.DataFrame(summaries)
    primary_metrics=['fiber_local_normalized_minimum','fiber_global_normalized_minimum',
                     'mlp_nuisance_cosine','test_accuracy']
    last_layer=int(frame.layer.max())
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,metric in zip(axes.flat,primary_metrics):
        part=summary[(summary.layer==last_layer)&(summary.metric==metric)]
        for relevance,group in part.groupby('relevance'):
            group=group.sort_values('gamma')
            ax.plot(group.gamma,group['mean'],marker='o',label=f'λ={relevance:g}')
            band=group[group.lower.notna()&group.upper.notna()]
            if not band.empty:
                ax.fill_between(band.gamma.to_numpy(float),band.lower.to_numpy(float),
                                band.upper.to_numpy(float),alpha=.12)
        ax.set_xscale('log',base=2)
        ax.set_xlabel('Output scale γ')
        ax.set_title(metric.replace('_',' '))
        ax.grid(alpha=.2)
    axes.flat[-1].legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle(f'Relevance dependence at last hidden layer (L{last_layer})')
    for suffix in ['png','pdf','svg']:
        fig.savefig(output/f'relevance_regime_map.{suffix}',dpi=220)
    plt.close(fig)
    result=dict(schema='feature-topology.relevance-statistics.v1',
                relevance_levels=levels,runs=int(frame.run_id.nunique()),rows=len(frame),
                bootstrap_repeats=repeats,replicate_unit='training seed',
                contrast='right relevance minus left relevance at fixed gamma and layer',
                multiple_comparisons='Descriptive pointwise intervals; no p-values or familywise claims.',
                scope='Matched-risk relevance dependence; geometry and predictive effects remain distinct.')
    atomic_json(output/'relevance_analysis_scope.json',result)
    return result
