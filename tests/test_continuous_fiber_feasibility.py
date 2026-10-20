import numpy as np
import pytest

torch=pytest.importorskip('torch')
from src.metrics.continuous_fibers import fit_circular_readout,features_and_phi_derivative
from src.models.mlp import MLP


def test_analytic_chain_rule_matches_autograd_at_regular_latent_points():
    torch.manual_seed(812)
    network=MLP(dimension=4,width=9,depth=3).double()
    latent=torch.tensor([[.31,.82],[2.12,4.48]],dtype=torch.float64,requires_grad=True)
    values,derivative=features_and_phi_derivative(network,latent,torch.eye(4))
    for sample in range(len(latent)):
        for feature in range(values.shape[1]):
            jac=torch.autograd.grad(values[sample,feature],latent,retain_graph=True)[0]
            assert derivative[sample,feature].item()==pytest.approx(jac[sample,1].item(),abs=1e-12)


def test_fixed_readout_recovers_a_scaled_rotated_circle_with_independent_task_coordinates():
    axis=np.arange(40)*2*np.pi/40
    theta,phi=np.meshgrid(axis,axis,indexing='ij')
    phi,theta=phi.ravel(),theta.ravel()
    features=np.column_stack([np.cos(theta),np.sin(theta),2*np.cos(phi),3*np.sin(phi)])
    readout,intercept,fit=fit_circular_readout(features,phi)
    assert np.allclose(readout,[[0,0,.5,0],[0,0,0,1/3]],atol=1e-9)
    assert np.linalg.norm(intercept)<1e-12
    assert fit['fit_rmse']<1e-9
