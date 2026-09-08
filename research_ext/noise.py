"""Held-out decoder fragility under controlled representation noise.

A clean-trained ridge decoder is tested at several noise levels. This measures
that decoder's robustness, not mutual information or existence of all decoders.
Noise realizations are paired across levels, not independent network replicates.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import file_digest, atomic_json, environment


def noise_curve(train_features, test_features, train_angles, test_angles, *,
                levels=(0., .0001, .001, .01, .1, 1.), repeats=5, seed=2026, alpha=.001) -> dict:
    from sklearn.linear_model import Ridge
    a, b = np.asarray(train_features, float), np.asarray(test_features, float)
    ta, tb = np.asarray(train_angles, float), np.asarray(test_angles, float)
    levels = np.asarray(levels, float)
    if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not a.shape[1]
            or len(a)<2 or len(b)<2 or ta.shape!=(len(a),) or tb.shape!=(len(b),)
            or not all(np.isfinite(v).all() for v in [a,b,ta,tb,levels])):
        raise ValueError('Finite train/test feature matrices and one angle per row are required')
    if (levels.ndim!=1 or not len(levels) or np.any(levels<0) or len(set(levels))!=len(levels)
            or repeats<1 or not np.isfinite(alpha) or alpha<=0):
        raise ValueError('Invalid noise levels, repetitions, or ridge regularization')
    center=a.mean(0)
    scale=float(np.sqrt(np.mean(np.sum((a-center)**2,axis=1))))
    if scale==0:
        return {'status':'constant_training_representation', 'training_rms':0., 'rows':[],
                'information_destruction_proved':False}
    x, y = (a-center)/scale, (b-center)/scale
    target=np.column_stack([np.cos(ta),np.sin(ta)])
    truth=np.column_stack([np.cos(tb),np.sin(tb)])
    decoder=Ridge(alpha=alpha).fit(x,target)
    rng=np.random.default_rng(seed)
    values={float(level):[] for level in levels}
    for _ in range(repeats):
        direction=rng.standard_normal(y.shape)/np.sqrt(y.shape[1])
        for level in levels:
            pred=decoder.predict(y+level*direction)
            lengths=np.linalg.norm(pred,axis=1)
            cosines=np.divide(np.sum(pred*truth,axis=1),lengths,out=np.zeros(len(pred)),where=lengths>0)
            # Clip only roundoff beyond the mathematical [-1,1] range.
            values[float(level)].append({'angular_cosine':float(np.mean(np.clip(cosines,-1,1))),
                'circular_mse':float(np.mean(np.sum((pred-truth)**2,axis=1)))})
    rows=[]
    for level, samples in sorted(values.items()):
        rows.append({'noise_to_training_rms':level,'coordinate_noise_sd':level*scale/np.sqrt(a.shape[1]),
            'mean_angular_cosine':float(np.mean([s['angular_cosine'] for s in samples])),
            'mean_circular_mse':float(np.mean([s['circular_mse'] for s in samples])),
            'noise_repetitions':samples})
    return {'status':'decoder_noise_diagnostic_completed', 'rows':rows,'training_rms':scale,
            'ridge_alpha':alpha,'noise_seed':seed,'noise_repeats':repeats,
            'training_rows':len(a),'test_rows':len(b),'feature_dimension':a.shape[1],
            'normalization':'Mean and global RMS fitted on training representations only',
            'perturbation':'Isotropic Gaussian noise at test time; expected RMS norm = level * training RMS',
            'decoder':'Ridge on clean training data predicting cosine/sine of the angle',
            'interpretation':'Noise realizations do not increase the number of independent training seeds.',
            'information_destruction_proved':False}


def checkpoint_noise(checkpoint: str | Path, output: str | Path, *, layer: int | None = None,
                     train_points=2000, test_points=1000, repeats=5) -> dict:
    import torch
    from src.training.train import build
    from src.data.torus import dataset
    checkpoint=Path(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=state['config']
    if config.get('manifold','torus')!='torus':
        raise ValueError('Circular angle diagnostic currently supports the torus, not a cylinder coordinate')
    if not 2<=train_points<=20000 or not 2<=test_points<=20000:
        raise ValueError('Probe sizes must lie in [2, 20000]')
    model=build(config,config['seed']);model.load_state_dict(state['model']);model.eval()
    depth=config.get('depth',4);layer=depth if layer is None else layer
    if not 1<=layer<=depth:raise ValueError('Invalid hidden layer')
    options={k:config[k] for k in ['dimension','manifold','swap','relevance','relevance_mode'] if k in config}
    x,_,angles_a,_=dataset(train_points,seed=4001,**options)
    y,_,angles_b,_=dataset(test_points,seed=4002,**options)
    # Exact duplicated latent samples across independent draws invalidate the split.
    if {tuple(v) for v in angles_a.tolist()} & {tuple(v) for v in angles_b.tolist()}:
        raise ValueError('Train/test latent overlap')
    def features(data):
        with torch.no_grad():
            return torch.cat([model.network.representations(part)[layer-1].cpu()
                              for part in data.split(512)]).numpy()
    nuisance=int(not config.get('swap',False))
    result=noise_curve(features(x),features(y),angles_a[:,nuisance].numpy(),angles_b[:,nuisance].numpy(),repeats=repeats)
    result.update(checkpoint_sha256=file_digest(checkpoint),step=state['step'],gamma=config['gamma'],
                  training_seed=config['seed'],hidden_layer=layer,probe_draw_seeds=[4001,4002],
                  factor_coordinate=nuisance,relevance=config.get('relevance',0.),environment=environment())
    atomic_json(output,result)
    return result
