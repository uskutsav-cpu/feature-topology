import pytest

pytest.importorskip('flint')
from flint import ctx
from src.metrics.joint_fiber_verifier import JointMaskFixedThetaNetwork
from src.metrics.fixed_fiber_verifier import replay_circle,verify_circle


def coupled_circle(maximum_uncertain=6):
    ctx.prec=128
    # Residual derivative is either (.6*cos(phi),0) or (0,.6*cos(phi)).
    # Its true norm is at most .6, while the independent rectangle allows .6*sqrt(2).
    layers=[([[0,0,1,0],[0,0,0,1],[0,0,0,1]],[2,2,0],True),
            ([[1,.6,-.6],[0,1,.6]],[0,0],False)]
    return JointMaskFixedThetaNetwork(layers,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                     [[1,0],[0,1]],theta_numerator=1,
                                     maximum_uncertain=maximum_uncertain)


def test_joint_activation_vertices_fix_the_known_componentwise_norm_gap():
    joint=coupled_circle();rectangle=coupled_circle(0)
    assert float(joint.bound('0'*14)[0])<.61
    assert float(rectangle.bound('0'*14)[0])>.8
    result=verify_circle(joint,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is True
    assert replay_circle(joint,result)
    limited=verify_circle(rectangle,maximum_boxes=2000,maximum_depth=14,seconds=60)
    assert limited['continuous_certificate'] is False


def test_joint_bounds_do_not_certify_the_two_to_one_circle_control():
    ctx.prec=128
    network=JointMaskFixedThetaNetwork([([[0,0,1,0],[0,0,0,1]],[2,2],True)],
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0],[0,1]],
        theta_numerator=3,winding=2)
    result=verify_circle(network,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is False
    assert result['status']=='fixed_witness_rejected'
