import copy
import pytest

pytest.importorskip('flint')
from flint import ctx
from src.metrics.fixed_fiber_verifier import FixedThetaIntervalNetwork,replay_circle,verify_circle


def network(projection=False,winding=1):
    ctx.prec=128
    return FixedThetaIntervalNetwork(
        [([[0,0,1,0],[0,0,0,0 if projection else 1]],[2,2],True)],
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,0],[0,1]],
        theta_numerator=1,winding=winding)


def test_fixed_smooth_circle_certifies_and_replays_without_upgrading_its_scope():
    net=network();result=verify_circle(net,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is True
    assert replay_circle(net,result)
    for key,value in [('theta_fraction',[3,8]),('scope','All theta values')]:
        damaged=copy.deepcopy(result);damaged[key]=value
        with pytest.raises(ValueError,match='scope changed'):replay_circle(net,damaged)
    damaged=copy.deepcopy(result);damaged['leaf_codes'].pop()
    with pytest.raises(ValueError,match='unresolved'):replay_circle(net,damaged)


@pytest.mark.parametrize('projection,winding',[(True,1),(False,2)])
def test_fixed_circle_projection_and_double_cover_are_not_certified(projection,winding):
    result=verify_circle(network(projection,winding),maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is False
    assert result['status']=='fixed_witness_rejected'
