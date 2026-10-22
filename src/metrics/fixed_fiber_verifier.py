"""Scoped, complete-circle verification at one exact task coordinate."""
from fractions import Fraction
import time

from flint import arb
from src.metrics.continuous_fiber_affine_verifier import AffineCircularIntervalNetwork
from src.metrics.continuous_fiber_verifier import angle_box

SCOPE='Specified theta nuisance circle only; no universal-theta or full-torus injectivity claim'


class FixedThetaIntervalNetwork(AffineCircularIntervalNetwork):
    def __init__(self,*args,theta_numerator,theta_denominator=8,**kwargs):
        super().__init__(*args,**kwargs)
        if theta_numerator not in {1,3,5,7} or theta_denominator!=8:
            raise ValueError('Expected a declared class-sector midpoint')
        self.theta_fraction=[theta_numerator,theta_denominator]

    def parameters(self,code,point=False):
        if any(char not in '01' for char in code):
            raise ValueError('Invalid binary circle interval')
        index=int(code,2) if code else 0
        theta=2*arb.pi()*self.theta_fraction[0]/self.theta_fraction[1]
        return theta,angle_box(index,len(code),point)


def check_circle_cover(leaves):
    ordered=sorted(leaves)
    if not ordered or len(set(ordered))!=len(ordered):
        raise ValueError('Empty or duplicate circle cover')
    for index,code in enumerate(ordered):
        if any(char not in '01' for char in code):
            raise ValueError('Invalid binary circle address')
        if index and code.startswith(ordered[index-1]):
            raise ValueError('Circle leaves overlap by ancestry')
    if sum((Fraction(1,2**len(code)) for code in leaves),Fraction(0))!=1:
        raise ValueError('Circle cover has unresolved intervals')
    return True


def verify_circle(network,maximum_boxes=20000,maximum_depth=18,seconds=3300):
    started=time.monotonic();pending=[''];leaves=[];checked=0;maximum=arb(0)
    threshold=2/arb.pi()
    common=dict(theta_fraction=network.theta_fraction,scope=SCOPE)
    while pending:
        if checked>=maximum_boxes or time.monotonic()-started>=seconds:break
        code=pending.pop();checked+=1
        upper,_,_=network.bound(code)
        if upper.is_finite() and upper<threshold:
            leaves.append(code);maximum=maximum.max(upper).upper();continue
        _,point_lower,strict=network.bound(code,point=True)
        if strict and point_lower.is_finite() and point_lower>threshold:
            return dict(**common,status='fixed_witness_rejected',continuous_certificate=False,
                        checked_boxes=checked,rejection_interval=code,
                        point_residual_lower=point_lower.str(60))
        if len(code)>=maximum_depth:
            pending.append(code);break
        pending.extend([code+'1',code+'0'])
    if pending:
        return dict(**common,status='inconclusive',continuous_certificate=False,
                    checked_boxes=checked,certified_leaves=len(leaves),unresolved_intervals=len(pending),
                    elapsed_seconds=time.monotonic()-started)
    check_circle_cover(leaves)
    epsilon=arb(maximum.str(60))
    factor=(1-epsilon*arb.pi()/2)/network.readout_norm
    if not epsilon.contains(maximum) or not epsilon<threshold or not factor>0:
        return dict(**common,status='inconclusive',continuous_certificate=False,
                    checked_boxes=checked,reason='No rigorously positive separation factor')
    return dict(**common,status='certified_fixed_nuisance_circle',continuous_certificate=True,
                checked_boxes=checked,leaf_codes=leaves,epsilon_enclosure=epsilon.str(60),
                chord_lower_factor_enclosure=factor.str(60),
                elapsed_seconds=time.monotonic()-started)


def replay_circle(network,result):
    if (result.get('status')!='certified_fixed_nuisance_circle'
            or result.get('continuous_certificate') is not True
            or result.get('theta_fraction')!=network.theta_fraction or result.get('scope')!=SCOPE):
        raise ValueError('Certificate status or task-coordinate scope changed')
    check_circle_cover(result['leaf_codes'])
    maximum=arb(0)
    for code in result['leaf_codes']:
        upper,_,_=network.bound(code)
        if not upper.is_finite() or not upper<2/arb.pi():
            raise ValueError('Circle interval failed standalone replay')
        maximum=maximum.max(upper).upper()
    epsilon=arb(result['epsilon_enclosure'])
    if not epsilon.contains(maximum) or not epsilon<2/arb.pi():
        raise ValueError('Replayed epsilon mismatch')
    factor=(1-epsilon*arb.pi()/2)/network.readout_norm
    claimed=arb(result['chord_lower_factor_enclosure'])
    if not factor>0 or not claimed>0 or not claimed.contains(factor):
        raise ValueError('Replayed separation factor mismatch')
    return True
