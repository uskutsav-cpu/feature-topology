"""Seed-level summaries for the recorded production noisy-decoding completion."""
import numpy as np
import pandas as pd

from research_ext.statistics import bootstrap_mean, paired_contrast


def summarize_noise(collection):
    records=collection['records']
    if (collection.get('complete_unique_models')!=305 or len(records)!=305
            or len({row['run_id'] for row in records})!=305):
        raise ValueError('Noise summaries require all 305 unique saved models')
    rows=[]
    for model in records:
        base={key:model[key] for key in ['run_id','gamma','seed','relevance','status',
            'checkpoint','checkpoint_sha256','selected_step','actual_training_loss',
            'constant_training_representation']}
        base.update(main_view='main' in model['roles'],relevance_view='relevance' in model['roles'])
        if model['status']=='evaluated':
            rows.extend(dict(**base,**point) for point in model['noise_rows'])
        else:
            rows.append(dict(**base,noise_level=None,training_rms=None,coordinate_noise_sd=None,
                             angular_cosine=None,angular_mae=None))
    frame=pd.DataFrame(rows)
    if frame.duplicated(['run_id','noise_level']).any():
        raise ValueError('Duplicate saved-model/noise-level records')
    summaries=[];contrasts=[]
    for view,flag in [('main','main_view'),('relevance','relevance_view')]:
        part=frame[frame[flag]&frame.status.eq('evaluated')]
        for (relevance,gamma,level),group in part.groupby(['relevance','gamma','noise_level']):
            for metric in ['angular_cosine','angular_mae']:
                if group.seed.duplicated().any():raise ValueError('Training seed duplicated in noise cell')
                result=bootstrap_mean(group.sort_values('seed')[metric].to_numpy(float),repeats=2000)
                summaries.append(dict(view=view,relevance=float(relevance),gamma=float(gamma),
                    noise_level=float(level),metric=metric,bootstrap_repeats=2000,**result))
    part=frame[frame.relevance_view&frame.status.eq('evaluated')]
    for (gamma,level),group in part.groupby(['gamma','noise_level']):
        for relevance in sorted(part.relevance.unique()):
            if relevance==0:continue
            for metric in ['angular_cosine','angular_mae']:
                left=group[group.relevance.eq(0)].set_index('seed')[metric].to_dict()
                right=group[group.relevance.eq(relevance)].set_index('seed')[metric].to_dict()
                contrasts.append(dict(gamma=float(gamma),noise_level=float(level),metric=metric,
                    relevance_left=0.,relevance_right=float(relevance),bootstrap_repeats=2000,
                    **paired_contrast(left,right,repeats=2000)))
    return dict(schema='feature-topology.production-noise-statistics.v1',
        unique_models=305,main_models=130,relevance_models=210,shared_models=35,
        evaluated_models=collection['evaluated_models'],unreached_models=collection['unreached_models'],
        noise_levels=collection['noise_levels'],replicate_unit='training seed',
        main_seed_count=10,relevance_seed_count=5,bootstrap_repeats=2000,
        interval_scope='Descriptive pointwise 95% percentile intervals; no familywise or hypothesis-test claims',
        method='Legacy level-adapted ridge refitted separately at every noise level; both training and test features receive relative noise',
        scope=collection['scope'],rows=rows,seed_confidence_intervals=summaries,
        relevance_paired_contrasts=contrasts)


def write_tables(result,output):
    pd.DataFrame(result['rows']).to_csv(output/'noise_metrics_long.csv',index=False)
    pd.DataFrame(result['seed_confidence_intervals']).to_csv(output/'noise_seed_confidence_intervals.csv',index=False)
    pd.DataFrame(result['relevance_paired_contrasts']).to_csv(output/'noise_relevance_paired_contrasts.csv',index=False)


def noise_figures(result,output,save_figure):
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(result['seed_confidence_intervals'])
    main=frame[frame.view.eq('main')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,metric,label in zip(axes,['angular_cosine','angular_mae'],
                              ['Held-out angular cosine','Held-out angular error (radians)']):
        for level,group in main[main.metric.eq(metric)].groupby('noise_level'):
            group=group.sort_values('gamma')
            ax.plot(group.gamma,group['mean'],'o-',ms=3,label=f'Relative noise {level:g}')
            band=group.dropna(subset=['lower','upper'])
            ax.fill_between(band.gamma.to_numpy(float),band.lower.to_numpy(float),band.upper.to_numpy(float),alpha=.12)
        ax.set(xscale='log',xlabel='Output scaling gamma',ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Production nuisance decoding at the first stored loss at or below 0.1')
    fig.supxlabel('Ten training seeds per gamma; bands are pointwise 95% seed-bootstrap intervals.\nA separate ridge probe is refitted at each noise level. Decoder failure does not prove information loss.',fontsize=9)
    save_figure(fig,output,'production_noise_main');plt.close(fig)
    relevance=frame[frame.view.eq('relevance')&frame.metric.eq('angular_cosine')&frame.noise_level.eq(.1)]
    levels=sorted(relevance.relevance.unique());gammas=sorted(relevance.gamma.unique())
    matrix=np.full((len(levels),len(gammas)),np.nan)
    for row in relevance.itertuples():matrix[levels.index(row.relevance),gammas.index(row.gamma)]=row.mean
    fig,ax=plt.subplots(figsize=(9,5.3),constrained_layout=True)
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='viridis',aspect='auto')
    for i in range(len(levels)):
        for j in range(len(gammas)):
            if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9,
                color='black' if matrix[i,j]>.3 else 'white')
    ax.set(xticks=range(len(gammas)),xticklabels=[f'{g:g}' for g in gammas],
        yticks=range(len(levels)),yticklabels=[f'{v:g}' for v in levels],xlabel='Output scaling gamma',
        ylabel='Task relevance lambda',title='Phi-coordinate decoding at relative noise 0.1')
    fig.colorbar(im,ax=ax,label='Mean held-out angular cosine')
    fig.supxlabel('Five paired training seeds per cell; all levels appear in the accompanying tables.\nAt positive lambda, phi is partly task relevant; these are not task-preserving nuisance fibers.',fontsize=9)
    save_figure(fig,output,'production_noise_relevance');plt.close(fig)
