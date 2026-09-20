"""Frozen cross-relevance summaries from condition-level matched-risk tables."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

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
    result=dict(schema='feature-topology.relevance-statistics.v1',
                relevance_levels=levels,runs=int(frame.run_id.nunique()),rows=len(frame),
                bootstrap_repeats=repeats,replicate_unit='training seed',
                contrast='right relevance minus left relevance at fixed gamma and layer',
                multiple_comparisons='Descriptive pointwise intervals; no p-values or familywise claims.',
                scope='Matched-risk relevance dependence; geometry and predictive effects remain distinct.')
    atomic_json(output/'relevance_analysis_scope.json',result)
    return result

