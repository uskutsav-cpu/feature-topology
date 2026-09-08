"""A REAL, tiny upstream training + analysis integration check, not production.

Uses the repository's actual MLP, torus generator and trainer. Only inexpensive
metrics are measured. Persistent homology and nonlinear probes are absent, never
filled with invented values. This deliberately does not recalibrate production.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json,read_json,original_fingerprint,environment
from .report import analyze
from .bridge import export_checkpoint
from .exact import point,Layer,polygon_certificate,verify_polygon_certificate


def _features(network, x):
    with torch.no_grad():
        return [h.cpu().numpy().astype(float) for h in network.representations(x)]


def _ntk(model, x):
    parameters=tuple(model.network.parameters())
    vectors=[]
    for value in x:
        logits=model(value.unsqueeze(0))[0]
        gradients=[]
        for c in range(len(logits)):
            g=torch.autograd.grad(logits[c],parameters,retain_graph=True,allow_unused=False)
            gradients.append(torch.cat([v.reshape(-1) for v in g]).detach().cpu().numpy())
        vectors.append(np.stack(gradients))
    j=np.stack(vectors)
    return np.einsum('ncp,mcp->nm',j,j).astype(float)


def run_smoke(output: str | Path) -> dict:
    try:
        from src.training.train import train,build,data_for,evaluate
        from src.data.torus import dataset,coordinates,geodesic
    except ModuleNotFoundError as exc:
        raise RuntimeError('Run smoke from the feature-topology checkout after applying this update') from exc
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge,LogisticRegression
    output=Path(output)
    options=dict(all_checkpoints=False,jacobian_points=8,ntk_points=4,ph_size=0,ph_repeats=0,
                 ph_maxdim=0,probe_train=128,probe_test=64,probe_iterations=0,
                 diagnostic_only=True,missing_metrics=['persistent_homology','nonlinear_probe'])
    profile=original_fingerprint(options)
    manifest=[]
    old_determinism=torch.are_deterministic_algorithms_enabled()
    old_threads=torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    try:
        for gamma in [.5,2.]:
            for seed in [0,1]:
                config=dict(gamma=gamma,seed=seed,dimension=16,width=8,depth=2,n_train=128,
                            n_validation=64,n_grid=64,max_steps=32,batch_size=32,eval_every=8,
                            target_loss=.6,lr=.02*gamma**2,threads=1,smoke_diagnostic=True)
                result=train(config,output/'runs')
                manifest.append({'run_id':result['run_id'],'gamma':gamma,'seed':seed,'status':result['status']})
                run=output/'runs'/result['run_id']
                if result['status']=='diverged':
                    raise RuntimeError('Tiny integration run diverged')
                model=build(config,seed)
                _,_,(gx,_,z,q)=data_for(config)
                initial=_features(model.network,gx)
                k0=_ntk(model,gx[:4])
                px,py,pz,_=dataset(128,seed=4001)
                tx,ty,tz,_=dataset(64,seed=4002)
                metrics=run/'metrics'/profile
                atomic_json(metrics/'options.json',options)
                for name in ['step_0000000','final']:
                    state=torch.load(run/(name+'.pt'),map_location='cpu',weights_only=True)
                    model.load_state_dict(state['model'])
                    current=_features(model.network,gx)
                    train_features=_features(model.network,px)
                    test_features=_features(model.network,tx)
                    test_loss,accuracy=evaluate(model,tx,ty)
                    k=_ntk(model,gx[:4])
                    row=dict(step=state['step'],gamma=gamma,seed=seed,test_loss=test_loss,
                             test_accuracy=accuracy,ntk_drift=float(np.linalg.norm(k-k0)/np.linalg.norm(k0)),layers=[])
                    for index,h in enumerate(current):
                        a=initial[index]-initial[index].mean(0);b=h-h.mean(0)
                        denominator=np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
                        cka=float(np.linalg.norm(a.T@b)**2/denominator) if denominator else None
                        scale=float(np.sqrt(np.mean(np.sum(b*b,axis=1))))
                        eig=np.maximum(np.linalg.eigvalsh(b.T@b),0)
                        probabilities=eig[eig>0]/eig.sum() if eig.sum() else np.array([])
                        rank=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if len(probabilities) else 0.
                        def on_latent(latent):
                            return model.network.representations(coordinates(latent)@q.T)[index]
                        j=torch.func.vmap(torch.func.jacfwd(on_latent))(z[:8]).detach().numpy()
                        singular=np.linalg.svd(j,compute_uv=False)[:,-1]
                        raw=float(np.quantile(singular,.01))
                        norms=np.linalg.norm(j,axis=1).mean(0)
                        scaler=StandardScaler().fit(train_features[index])
                        train_h=scaler.transform(train_features[index]);test_h=scaler.transform(test_features[index])
                        targets=np.column_stack([np.cos(pz[:,1].numpy()),np.sin(pz[:,1].numpy())])
                        true=np.column_stack([np.cos(tz[:,1].numpy()),np.sin(tz[:,1].numpy())])
                        prediction=Ridge(alpha=.001).fit(train_h,targets).predict(test_h)
                        cosine=float(np.mean(np.sum(prediction*true,axis=1)/np.maximum(np.linalg.norm(prediction,axis=1),1e-12)))
                        task_accuracy=float(LogisticRegression(max_iter=300,random_state=seed).fit(train_h,py.numpy()).score(test_h,ty.numpy()))
                        i,jj=np.triu_indices(len(h),1)
                        distance=geodesic(z[i].numpy(),z[jj].numpy())
                        eligible=distance>.5
                        ratios=np.linalg.norm(h[i[eligible]]-h[jj[eligible]],axis=1)/distance[eligible]
                        global_q=float(np.quantile(ratios,.01))
                        row['layers'].append(dict(layer=index+1,cka_drift=None if cka is None else 1-cka,
                            effective_rank=rank,scale=scale,jacobian=dict(q01=raw,normalized_q01=raw/scale if scale else None,
                            task_norm=float(norms[0]),nuisance_norm=float(norms[1])),
                            global_margin=dict(q01=global_q,normalized_q01=global_q/scale if scale else None),
                            probes=dict(linear=dict(angular_cosine=cosine,task_accuracy=task_accuracy)),persistence=[]))
                    atomic_json(metrics/(name+'.json'),row)
        atomic_json(output/'manifest.json',manifest)
        analysis=analyze([output],output/'analysis',target=1.3,profile=profile,gammas=[.5,2.],seeds=[0,1],
                         repeats=100,transition_repeats=10,make_plots=False)
        first=output/'runs'/manifest[0]['run_id']/'final.pt'
        exported=export_checkpoint(first,hidden_layers=1)
        layers=[Layer.from_dict(v) for v in exported['layers']]
        vertices=[point([x,y]+[0]*14) for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        cert=polygon_certificate(vertices,layers,max_segments=128)
        cert['provenance']=exported['provenance']
        if not verify_polygon_certificate(cert):raise RuntimeError('Trained-network exact replay failed')
        atomic_json(output/'trained_polygon_certificate.json',cert)
        status={'status':'smoke_completed','actual_training_runs':len(manifest),
                'gamma_values':[.5,2.],'seeds':[0,1],'steps_budget_per_run':32,
                'analysis':analysis,'trained_polygon_exact_replay_passed':True,
                'lean_verified':False,'production_study_completed':False,
                'note':'Tiny integration experiment; does not establish the production hypotheses.'}
        atomic_json(output/'environment.json',environment())
        atomic_json(output/'smoke_result.json',status)
        return status
    finally:
        torch.use_deterministic_algorithms(old_determinism)
        torch.set_num_threads(old_threads)
