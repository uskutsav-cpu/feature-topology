import pytest

pytest.importorskip('flint')
from flint import ctx
from src.metrics.continuous_fiber_affine_verifier import AffineCircularIntervalNetwork
from src.metrics.continuous_fiber_verifier import replay_certificate,verify_witness


def test_strictly_active_relu_circle_control_certifies_and_replays():
    ctx.prec=128
    layers=[([[0,0,1,0],[0,0,0,1]],[2,2],True)]
    net=AffineCircularIntervalNetwork(layers,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0],[0,1]])
    result=verify_witness(net,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is True
    assert replay_certificate(net,result)


def test_affine_enclosures_do_not_certify_a_folded_relu_circle():
    ctx.prec=128
    layers=[([[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]],[0,0,0,0],True),
            ([[1,1,0,0],[0,0,1,1]],[0,0],False)]
    net=AffineCircularIntervalNetwork(layers,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0],[0,1]])
    result=verify_witness(net,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is False


def test_affine_relaxation_preserves_cancellation_through_stable_layers():
    ctx.prec=128
    # A large canceling task-coordinate component should not inflate later
    # activation ranges. This independently checks the purpose of the enclosure.
    layers=[([[100,0,1,0],[100,0,0,0],[0,0,0,1]],[200,200,2],True),
            ([[1,-1,0],[0,0,1]],[2,0],True)]
    net=AffineCircularIntervalNetwork(layers,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0],[0,1]])
    result=verify_witness(net,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is True
    assert replay_certificate(net,result)
