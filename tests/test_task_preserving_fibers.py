import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_task_preserving_fibers import task_plan
from src.metrics.task_preserving_fibers import (adversarial_pairs,chain_rule,dense_pairs,
    embedded,evaluate_family,latent_pair)
from src.models.mlp import MLP,ScaledModel


def test_task_plan_covers_exact_original_relevance_grid():
    rows,protocol=task_plan(Path(__file__).resolve().parents[1])
    assert len(rows)==210 and sum(row['relevance']==0 for row in rows)==35
    assert all('relevance' in row['roles'] for row in rows)


def test_mixed_task_derivative_matches_autograd_and_has_correct_unit_speeds():
    torch.manual_seed(19);network=MLP(dimension=4,width=12,depth=2).double().requires_grad_(False)
    q=torch.eye(4,dtype=torch.float64);base=torch.tensor([.17,.42],dtype=torch.float64)
    phi=torch.tensor([.31,2.23],dtype=torch.float64,requires_grad=True);relevance=.5
    latent=latent_pair(base,phi,relevance,True)
    h,d,flat_speed,input_speed,_=chain_rule(network,latent,q,relevance,True)
    observed=torch.stack([torch.autograd.grad(h[:,i].sum(),phi,retain_graph=True)[0] for i in range(12)],dim=1)
    torch.testing.assert_close(d,observed,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(flat_speed,(1+relevance**2*phi.cos().square()).sqrt())
    torch.testing.assert_close(input_speed,flat_speed)
    torch.testing.assert_close(latent[:,0]+relevance*latent[:,1].sin(),base)
    control=latent_pair(base,phi,relevance,False)
    assert not torch.allclose(control[:,0]+relevance*control[:,1].sin(),base)


def test_dense_separation_distinguishes_identity_circle_and_collapsed_control():
    side=12;phi=np.arange(side)*2*np.pi/side
    z=np.stack((np.zeros_like(phi),phi),axis=-1)[None]
    h=np.stack((np.cos(phi),np.sin(phi)),axis=-1)[None]
    result=dense_pairs(h,z,1.,1e-10)
    assert result['pairs']==66 and result['exact_float_collision_pairs']==0
    assert result['phi_arc']['minimum']==pytest.approx(2/np.pi)
    assert result['flat_torus_ambient']==result['phi_arc']
    collapsed=dense_pairs(np.zeros_like(h),z,0.,1e-10)
    assert collapsed['exact_float_collision_pairs']==collapsed['tolerance_collision_pairs']==66
    assert collapsed['phi_arc']['normalized_minimum'] is None


def test_adversarial_candidates_stay_on_true_fibers_and_preserve_separated_arc():
    torch.manual_seed(4);network=MLP(dimension=4,width=8,depth=1).double().requires_grad_(False)
    q=torch.eye(4,dtype=torch.float64)
    settings=dict(starts=4,steps=3,learning_rate=.03,seed=1729,
        minimum_phi_arc=math.pi/8,maximum_phi_arc=math.pi)
    result=adversarial_pairs(network,q,.5,True,2.,settings)
    best=result['best_candidate'];left,right=np.asarray(best['latent_pair'])
    assert result['evaluated_candidates']==16
    assert math.pi/8<=best['phi_arc']<=math.pi
    assert best['task_coordinate_wrapped_difference']<1e-12
    assert left[0]+.5*np.sin(left[1])==pytest.approx(right[0]+.5*np.sin(right[1]))
    z=torch.tensor(np.stack((left,right)),dtype=torch.float64)
    h=network.representations(embedded(z,q))[-1].detach().numpy()
    assert best['representation_distance']==pytest.approx(np.linalg.norm(h[0]-h[1]))
    assert best['normalized_phi_arc_ratio']==pytest.approx(best['phi_arc_ratio']/2)
    assert adversarial_pairs(network,q,.5,True,0.,settings)['status']=='constant_grid_scale'


def test_lambda_zero_task_and_control_searches_coincide_exactly():
    torch.manual_seed(2);network=MLP(dimension=4,width=8,depth=1).double().requires_grad_(False)
    q=torch.eye(4,dtype=torch.float64)
    settings=dict(starts=4,steps=2,learning_rate=.03,seed=1729,
        minimum_phi_arc=math.pi/8,maximum_phi_arc=math.pi)
    assert adversarial_pairs(network,q,0.,True,1.,settings)==adversarial_pairs(network,q,0.,False,1.,settings)


def test_family_evaluation_accounts_all_grid_points_pairs_and_true_task_coordinates():
    torch.manual_seed(3)
    model=ScaledModel(MLP(dimension=4,width=8,depth=1)).double().requires_grad_(False)
    q=torch.eye(4,dtype=torch.float64)
    protocol=dict(grid_side=8,collision_tolerance=1e-10,
        adversarial_search=dict(starts=3,steps=1,learning_rate=.03,seed=1729,
            minimum_phi_arc=math.pi/8,maximum_phi_arc=math.pi))
    result=evaluate_family(model,q,.5,True,protocol)
    assert result['grid_points']==64 and result['dense_pairs']['pairs']==224
    assert result['task_coordinate_maximum_error']<1e-12
    assert result['adversarial_search']['evaluated_candidates']==6
    assert result['classifier_grid_accuracy']==.25
