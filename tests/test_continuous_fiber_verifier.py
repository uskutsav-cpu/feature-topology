import copy

import pytest

pytest.importorskip('flint')
from flint import ctx
from src.metrics.continuous_fiber_verifier import (CircularIntervalNetwork,check_cover,
                                                  replay_certificate,verify_witness)


def control(winding=1,projection=False):
    ctx.prec=128
    layers=[([[0,0,1,0],[0,0,0,0 if projection else 1]],[0,0],False)]
    return CircularIntervalNetwork(layers,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                   [[1,0],[0,1]],winding=winding)


def test_complete_continuous_circle_control_certifies_and_replays():
    network=control()
    result=verify_witness(network,maximum_boxes=2000,seconds=60)
    assert result['continuous_certificate'] is True
    assert replay_certificate(network,result)
    damaged=copy.deepcopy(result)
    damaged['leaf_codes'].pop()
    with pytest.raises(ValueError,match='unresolved domain'):
        replay_certificate(network,damaged)
    damaged=copy.deepcopy(result)
    damaged['chord_lower_factor_enclosure']='1000'
    with pytest.raises(ValueError,match='separation factor'):
        replay_certificate(network,damaged)


@pytest.mark.parametrize('winding,projection',[(2,False),(1,True)])
def test_noninjective_controls_cannot_receive_a_positive_certificate(winding,projection):
    result=verify_witness(control(winding,projection),maximum_boxes=2000,seconds=60)
    assert result['status']=='fixed_witness_rejected'
    assert result['continuous_certificate'] is False


def test_cover_rejects_duplicates_overlapping_ancestors_and_missing_regions():
    assert check_cover(['0','1','2','3'])
    for leaves in [['0','1','2'],['','0'],['0','1','2','3','3']]:
        with pytest.raises(ValueError):
            check_cover(leaves)
