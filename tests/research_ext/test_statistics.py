import numpy as np
import pytest
from research_ext.statistics import bootstrap_mean,paired_contrast,model_comparison,crossing_interval,event_order

def test_bootstrap_reproducible():
    assert bootstrap_mean([1,2,3],repeats=99)==bootstrap_mean([1,2,3],repeats=99)

def test_single_seed_not_confidence():
    result=bootstrap_mean([4])
    assert result['lower'] is None and result['status']=='insufficient_seeds'

def test_paired_constant_difference():
    result=paired_contrast({0:100,1:-100,2:3},{0:102,1:-98,3:20},repeats=100)
    assert result['lower']==result['upper']==result['mean']==2
    assert result['left_only']==[2] and result['right_only']==[3]

@pytest.mark.parametrize('bad', [[1,float('nan')],[[1,2]], [float('inf')]])
def test_bad_bootstrap(bad):
    with pytest.raises(ValueError): bootstrap_mean(bad)

def test_pilot_does_not_invent_breakpoint():
    result=model_comparison([.03,.5,1,16,128],np.ones((3,5)),repeats=2)
    assert result['status']=='insufficient_data'

def test_constant_curve_prefers_constant():
    g=2.**np.arange(-5,8)
    result=model_comparison(g,np.ones((3,13)),repeats=10)
    assert result['aicc_winner']=='constant'
    assert result['conditional_hinge_gamma_interval'] is None

def test_clear_hinge_selected():
    x=np.arange(-5,8,dtype=float)
    y=1+.05*x+2*np.maximum(x-1,0)
    result=model_comparison(2.**x,np.stack([y,y+.001,y-.001]),repeats=10)
    assert result['aicc_winner']=='hinge'
    assert result['conditional_hinge_gamma_interval']==[2.,2.]
    assert result['status']=='exploratory'

@pytest.mark.parametrize('values,status,lower,upper', [
    ([0,.1,.8],'observed_bracket',10,20),
    ([.8,.9,.9],'left_censored',None,0),
    ([0,.1,.2],'right_censored',20,None),
    ([None,None,None],'unobserved',None,None),
    ([0,None,.8],'observed_bracket',0,20),
])
def test_crossing(values,status,lower,upper):
    r=crossing_interval([0,10,20],values,threshold=.5,direction='above')
    assert (r['status'],r['lower_step'],r['upper_step'])==(status,lower,upper)

def test_censored_order_not_imputed():
    a=crossing_interval([0,10,20],[0,.8,.9],threshold=.5,direction='above')
    b=crossing_interval([0,10,20],[0,.1,.2],threshold=.5,direction='above')
    assert event_order(a,b)=='a_before_b'
    assert event_order(a,a)=='unresolved'

def test_sustained_requires_consecutive_measurements():
    r=crossing_interval([0,1,2,3],[0,.8,None,.9],threshold=.5,direction='above',consecutive=2)
    assert r['upper_step'] is None

def test_unsorted_steps_rejected():
    with pytest.raises(ValueError): crossing_interval([10,0],[0,1],threshold=.5,direction='above')


def test_missing_sample_does_not_turn_previous_hit_into_nonhit():
    r=crossing_interval([0,1,2,3,4],[0,.8,None,.8,.9],threshold=.5,direction='above',consecutive=2)
    assert r['lower_step']==0 and r['upper_step']==3 and r['confirmed_at_step']==4

def test_unconfirmed_hit_is_not_claimed_as_right_censored_no_crossing():
    r=crossing_interval([0,1],[0,.8],threshold=.5,direction='above',consecutive=2)
    assert r['status']=='unconfirmed'
    assert event_order(r,{'status':'observed_bracket','lower_step':3,'upper_step':4})=='unresolved'
