"""Paired training-seed summaries for finite task-preserving fiber diagnostics."""
import numpy as np
import pandas as pd

from research_ext.statistics import bootstrap_mean,paired_contrast


METRICS=['representation_rms','local_per_phi_minimum','local_unit_torus_minimum',
    'local_unit_torus_normalized_minimum','local_unit_torus_normalized_q01',
    'local_unit_input_normalized_minimum','dense_phi_minimum','dense_phi_normalized_minimum',
    'dense_phi_normalized_q01','dense_ambient_minimum','dense_ambient_normalized_minimum',
    'search_phi_normalized_best','exact_float_collision_pairs','tolerance_collision_pairs','classifier_grid_accuracy']


def metric_values(family):
    dense=family['dense_pairs'];search=family['adversarial_search']['best_candidate']
    return dict(representation_rms=family['centered_representation_rms'],
        local_per_phi_minimum=family['local_per_phi']['minimum'],
        local_unit_torus_minimum=family['local_per_unit_flat_torus_tangent']['minimum'],
        local_unit_torus_normalized_minimum=family['local_per_unit_flat_torus_tangent']['normalized_minimum'],
        local_unit_torus_normalized_q01=family['local_per_unit_flat_torus_tangent']['normalized_q01'],
        local_unit_input_normalized_minimum=family['local_per_unit_embedded_input_tangent']['normalized_minimum'],
        dense_phi_minimum=dense['phi_arc']['minimum'],dense_phi_normalized_minimum=dense['phi_arc']['normalized_minimum'],
        dense_phi_normalized_q01=dense['phi_arc']['normalized_q01'],
        dense_ambient_minimum=dense['flat_torus_ambient']['minimum'],
        dense_ambient_normalized_minimum=dense['flat_torus_ambient']['normalized_minimum'],
        search_phi_normalized_best=search['normalized_phi_arc_ratio'] if search else None,
        exact_float_collision_pairs=dense['exact_float_collision_pairs'],
        tolerance_collision_pairs=dense['tolerance_collision_pairs'],classifier_grid_accuracy=family['classifier_grid_accuracy'])


def summarize_task_fibers(collection):
    records=collection['records']
    if (collection.get('complete_models')!=210 or len(records)!=210
            or len({r['run_id'] for r in records})!=210):raise ValueError('Task-fiber summaries require all 210 models')
    rows=[]
    for model in records:
        base={key:model[key] for key in ['run_id','gamma','seed','relevance','status','checkpoint_sha256',
            'selected_step','actual_training_loss']}
        for name in ['fixed_u_task_preserving','fixed_theta_original_factor_control']:
            family=next((f for f in model['families'] if f['family']==name),None)
            rows.append(dict(**base,family=name,**(metric_values(family) if family else {key:None for key in METRICS})))
    frame=pd.DataFrame(rows)
    if frame.duplicated(['relevance','gamma','seed','family']).any():raise ValueError('Task-fiber paired seed duplicated')
    summaries=[];family_contrasts=[];relevance_contrasts=[]
    for (level,gamma,family),group in frame.groupby(['relevance','gamma','family']):
        for metric in METRICS:
            values=group.sort_values('seed')[metric].dropna().to_numpy(float)
            summaries.append(dict(relevance=float(level),gamma=float(gamma),family=family,metric=metric,
                **bootstrap_mean(values,repeats=2000)))
    for (level,gamma),group in frame.groupby(['relevance','gamma']):
        for metric in METRICS:
            left=group[group.family.eq('fixed_theta_original_factor_control')].set_index('seed')[metric].dropna().to_dict()
            right=group[group.family.eq('fixed_u_task_preserving')].set_index('seed')[metric].dropna().to_dict()
            family_contrasts.append(dict(relevance=float(level),gamma=float(gamma),metric=metric,
                comparison='fixed-u minus fixed-theta',**paired_contrast(left,right,repeats=2000)))
    for (gamma,family),group in frame.groupby(['gamma','family']):
        for level in sorted(frame.relevance.unique()):
            if level==0:continue
            for metric in METRICS:
                left=group[group.relevance.eq(0)].set_index('seed')[metric].dropna().to_dict()
                right=group[group.relevance.eq(level)].set_index('seed')[metric].dropna().to_dict()
                relevance_contrasts.append(dict(gamma=float(gamma),family=family,metric=metric,
                    relevance_left=0.,relevance_right=float(level),**paired_contrast(left,right,repeats=2000)))
    return dict(schema='feature-topology.task-preserving-fiber-statistics.v1',models=210,
        evaluated_models=collection['evaluated_models'],unreached_models=collection['unreached_models'],
        replicate_unit='training seed',training_seeds=list(range(5)),bootstrap_repeats=2000,
        interval_scope='Descriptive pointwise 95% percentile seed-bootstrap intervals; no familywise or hypothesis-test claims',
        scope=collection['scope'],rows=rows,seed_confidence_intervals=summaries,
        family_paired_contrasts=family_contrasts,relevance_paired_contrasts=relevance_contrasts)


def write_tables(result,output):
    for field,name in [('rows','task_fiber_metrics_long'),('seed_confidence_intervals','task_fiber_seed_confidence_intervals'),
        ('family_paired_contrasts','task_fiber_family_paired_contrasts'),('relevance_paired_contrasts','task_fiber_relevance_paired_contrasts')]:
        pd.DataFrame(result[field]).to_csv(output/(name+'.csv'),index=False)


def task_fiber_figures(result,output,save_figure):
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(result['seed_confidence_intervals'])
    metrics=['local_unit_torus_normalized_minimum','dense_phi_normalized_minimum',
        'search_phi_normalized_best','classifier_grid_accuracy']
    labels=['Local unit-tangent minimum / RMS','Grid phi-arc minimum / RMS',
        'Best searched phi-arc ratio / RMS','Finite-grid classifier accuracy']
    fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
    for i,(family,title) in enumerate([('fixed_u_task_preserving','Fixed task coordinate u'),
                                       ('fixed_theta_original_factor_control','Original fixed-theta control')]):
        for j,(metric,label) in enumerate(zip(metrics,labels)):
            ax=axes[i,j]
            for level,group in frame[frame.family.eq(family)&frame.metric.eq(metric)].groupby('relevance'):
                group=group.sort_values('gamma');ax.plot(group.gamma,group['mean'],'o-',ms=3,label=f'lambda {level:g}')
                band=group.dropna(subset=['lower','upper'])
                ax.fill_between(band.gamma.to_numpy(float),band.lower.to_numpy(float),band.upper.to_numpy(float),alpha=.12)
            ax.set(xscale='log',xlabel='Output scaling gamma',ylabel=label,title=title if j==0 else '')
            if metric!='classifier_grid_accuracy':
                values=frame[frame.family.eq(family)&frame.metric.eq(metric)][['mean','lower','upper']].to_numpy(float)
                if np.all(values[np.isfinite(values)]>0):ax.set_yscale('log')
                else:
                    ax.set_yscale('symlog',linthresh=1e-6);ax.set_ylim(bottom=0)
            else:ax.set_ylim(0,1)
            ax.grid(alpha=.2)
    axes[0,0].legend(loc='lower left',ncol=2,fontsize=8)
    fig.suptitle('Task-preserving circles versus original-factor controls: five paired training seeds')
    fig.supxlabel('Bands are pointwise 95% seed-bootstrap intervals. All margins are finite sampled/search evidence.\nThe separated-pair search yields an upper bound on an infimum; none of these margins is a continuum certificate.',fontsize=9)
    save_figure(fig,output,'task_preserving_fiber_audit');plt.close(fig)
