"""Finite fixed-task-coordinate diagnostics; never continuum certificates."""
import math
import numpy as np
import torch


def latent_pair(base,phi,relevance,task_preserving):
    theta=base-relevance*phi.sin() if task_preserving else base+torch.zeros_like(phi)
    return torch.stack((theta,phi),dim=-1)


def embedded(latent,q):
    theta,phi=latent.unbind(-1)
    return torch.stack((theta.cos(),theta.sin(),phi.cos(),phi.sin()),dim=-1)@q.T


def chain_rule(network,latent,q,relevance,task_preserving):
    theta,phi=latent.unbind(-1)
    theta_speed=-relevance*phi.cos() if task_preserving else torch.zeros_like(phi)
    tangent=torch.stack((-theta.sin()*theta_speed,theta.cos()*theta_speed,
                          -phi.sin(),phi.cos()),dim=-1)@q.T
    input_speed=tangent.norm(dim=-1)
    flat_speed=(1+theta_speed.square()).sqrt()
    x=embedded(latent,q);zero_count=0
    for layer in network.hidden:
        before=layer(x)
        zero_count+=int((before==0).sum())
        tangent=(tangent@layer.weight.T)*(before>0)
        x=torch.relu(before)
    return x,tangent,flat_speed,input_speed,zero_count


def summarize_norms(values,scale):
    values=np.asarray(values,dtype=float)
    return dict(minimum=float(values.min()),q01=float(np.quantile(values,.01)),
        median=float(np.median(values)),normalized_minimum=float(values.min()/scale) if scale>0 else None,
        normalized_q01=float(np.quantile(values,.01)/scale) if scale>0 else None)


def arc_distance(left,right):
    delta=np.abs(left-right)%(2*np.pi)
    return np.minimum(delta,2*np.pi-delta)


def dense_pairs(h,latent,scale,tolerance):
    side=h.shape[1];upper=np.triu_indices(side,1)
    phi_ratios=[];ambient_ratios=[];minimum_distance=math.inf;exact=0;tolerant=0;best=None
    for index,(fiber,z) in enumerate(zip(h,latent)):
        distances=np.linalg.norm(fiber[:,None,:]-fiber[None,:,:],axis=-1)[upper]
        dphi=arc_distance(z[:,1,None],z[None,:,1])[upper]
        dtheta=arc_distance(z[:,0,None],z[None,:,0])[upper]
        ambient=np.hypot(dtheta,dphi)
        if np.any(dphi<=0):raise ValueError('Distinct fiber points need distinct phi coordinates')
        ratios=distances/dphi
        k=int(np.argmin(ratios))
        if best is None or ratios[k]<best['phi_arc_ratio']:
            best=dict(base_index=index,pair_indices=[int(upper[0][k]),int(upper[1][k])],
                latent_pair=[z[upper[0][k]].tolist(),z[upper[1][k]].tolist()],
                phi_arc_ratio=float(ratios[k]),representation_distance=float(distances[k]))
        phi_ratios.append(ratios);ambient_ratios.append(distances/ambient)
        minimum_distance=min(minimum_distance,float(distances.min()))
        exact+=int(np.sum(np.all(fiber[upper[0]]==fiber[upper[1]],axis=1)))
        tolerant+=int(np.sum(distances<=tolerance))
    return dict(phi_arc=summarize_norms(np.concatenate(phi_ratios),scale),
        flat_torus_ambient=summarize_norms(np.concatenate(ambient_ratios),scale),
        pairs=int(len(h)*len(upper[0])),minimum_representation_distance=minimum_distance,
        exact_float_collision_pairs=exact,tolerance_collision_pairs=tolerant,
        collision_tolerance=tolerance,minimum_phi_ratio_witness=best,
        qualification='Finite grid only; no continuum lower bound or real-valued collision certificate')


def adversarial_pairs(network,q,relevance,task_preserving,scale,settings):
    if scale==0:return dict(status='constant_grid_scale',evaluated_candidates=0,best_candidate=None)
    rng=np.random.default_rng(settings['seed']);starts=settings['starts']
    base=rng.uniform(0,2*np.pi,starts);phi=rng.uniform(0,2*np.pi,starts)
    low,high=settings['minimum_phi_arc'],settings['maximum_phi_arc']
    arc=rng.uniform(low,high,starts);p=(arc-low)/(high-low)
    parameters=torch.nn.Parameter(torch.tensor(np.column_stack((base,phi,np.log(p/(1-p)))),dtype=torch.float64))
    optimizer=torch.optim.Adam([parameters],lr=settings['learning_rate'])
    best=None
    for step in range(settings['steps']+1):
        b,a,v=parameters.unbind(-1);delta=low+(high-low)*v.sigmoid()
        left=latent_pair(b,a,relevance,task_preserving)
        right=latent_pair(b,a+delta,relevance,task_preserving)
        h1=network.representations(embedded(left,q))[-1]
        h2=network.representations(embedded(right,q))[-1]
        distances=(h1-h2).norm(dim=-1);ratios=distances/delta
        if not torch.isfinite(ratios).all():raise ValueError('Nonfinite adversarial candidate')
        k=int(ratios.detach().argmin());score=float(ratios[k].detach())
        if best is None or score<best['phi_arc_ratio']:
            z1,z2=left[k].detach().numpy(),right[k].detach().numpy()
            t1=z1[0]+relevance*np.sin(z1[1]);t2=z2[0]+relevance*np.sin(z2[1])
            ambient=float(np.linalg.norm(arc_distance(z1,z2)))
            best=dict(update=step,start=k,base=float(b[k].detach()),latent_pair=[z1.tolist(),z2.tolist()],
                phi_arc=float(delta[k].detach()),representation_distance=float(distances[k].detach()),
                phi_arc_ratio=score,normalized_phi_arc_ratio=score/scale,
                flat_torus_ambient_ratio=float(distances[k].detach())/ambient,
                normalized_flat_torus_ambient_ratio=float(distances[k].detach())/(ambient*scale),
                task_coordinate_pair=[float(t1),float(t2)],task_coordinate_wrapped_difference=float(arc_distance(t1,t2)))
        if step<settings['steps']:
            optimizer.zero_grad();(ratios/scale).mean().backward();optimizer.step()
    return dict(status='fixed_budget_complete',evaluated_candidates=starts*(settings['steps']+1),
        best_candidate=best,qualification='Smallest evaluated separated-pair ratio; upper bound on an infimum, not a certificate')


def evaluate_family(model,q,relevance,task_preserving,protocol):
    side=protocol['grid_side'];angles=(torch.arange(side,dtype=torch.float64)+.5)*(2*math.pi/side)
    base,phi=torch.meshgrid(angles,angles,indexing='ij')
    latent=latent_pair(base.flatten(),phi.flatten(),relevance,task_preserving)
    features=[];derivatives=[];flat_speeds=[];input_speeds=[];correct=0;zeros=0
    with torch.no_grad():
        for z in latent.split(1024):
            h,d,speed,input_speed,count=chain_rule(model.network,z,q,relevance,task_preserving)
            features.append(h.numpy());derivatives.append(d.norm(dim=-1).numpy())
            flat_speeds.append(speed.numpy());input_speeds.append(input_speed.numpy());zeros+=count
            labels=((z[:,0]+relevance*z[:,1].sin())%(2*math.pi)/(math.pi/2)).long()
            correct+=int((model(embedded(z,q)).argmax(dim=-1)==labels).sum())
    h=np.concatenate(features);norms=np.concatenate(derivatives)
    scale=float(np.sqrt(np.mean(np.sum((h-h.mean(axis=0))**2,axis=1))))
    z=latent.numpy().reshape(side,side,2)
    u=(z[:,:,0]+relevance*np.sin(z[:,:,1]))%(2*np.pi)
    if task_preserving:
        error=float(arc_distance(u,base.numpy()).max())
        if error>1e-12:raise ValueError('Fixed-u parameterization changed the task coordinate')
        labels=(u/(np.pi/2)).astype(int)
        if np.any(labels!=labels[:,:1]):raise ValueError('Fixed-u grid changed class within a circle')
    else:error=None
    return dict(family='fixed_u_task_preserving' if task_preserving else 'fixed_theta_original_factor_control',
        grid_side=side,grid_points=side*side,centered_representation_rms=scale,
        local_per_phi=summarize_norms(norms,scale),
        local_per_unit_flat_torus_tangent=summarize_norms(norms/np.concatenate(flat_speeds),scale),
        local_per_unit_embedded_input_tangent=summarize_norms(norms/np.concatenate(input_speeds),scale),
        exact_zero_preactivations=zeros,task_coordinate_maximum_error=error,
        classifier_grid_accuracy=correct/(side*side),
        dense_pairs=dense_pairs(h.reshape(side,side,-1),z,scale,protocol['collision_tolerance']),
        adversarial_search=adversarial_pairs(model.network,q,relevance,task_preserving,scale,protocol['adversarial_search']))
