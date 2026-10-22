"""Conservative Arb-ball verifier for a fixed circular nuisance readout.

The source network is interpreted over the reals with exact dyadic coefficients.
This is not a floating-point grid test and is not covered by the Lean lemmas.
The verifier can be inconclusive even for an injective map.
"""
from fractions import Fraction
import time

from flint import arb, arb_mat, ctx


def ball_matrix(values):
    return arb_mat([[arb(float(value)) for value in row] for row in values])


def decode_box(code):
    i=j=0
    for char in code:
        if char not in '0123':
            raise ValueError('Invalid quadtree address')
        child=int(char)
        i,j=2*i+child//2,2*j+child%2
    return i,j,len(code)


def angle_box(index, depth, point=False):
    denominator=1 << (depth+1)
    midpoint=arb(2*index+1)/denominator
    radius=arb(0) if point else arb(1)/denominator
    # Both midpoint and radius are exact dyadics before multiplication by pi.
    return arb(midpoint, radius)*2*arb.pi()


def check_cover(leaves):
    ordered=sorted(leaves)
    if not ordered or len(set(ordered))!=len(ordered):
        raise ValueError('Empty or duplicate certificate cover')
    for index,code in enumerate(ordered):
        decode_box(code)
        if index and code.startswith(ordered[index-1]):
            raise ValueError('Certificate leaves overlap by ancestry')
    if sum((Fraction(1,4**len(code)) for code in leaves),Fraction(0))!=1:
        raise ValueError('Certificate cover has an unresolved domain region')
    return True


class CircularIntervalNetwork:
    def __init__(self,layers,embedding,readout,winding=1):
        self.layers=[(ball_matrix(weight),arb_mat([[arb(float(v))] for v in bias]),relu)
                     for weight,bias,relu in layers]
        if not self.layers or winding<1 or int(winding)!=winding:
            raise ValueError('Nonempty network and positive integer control winding required')
        self.readout=ball_matrix(readout)
        self.winding=int(winding)
        weight,bias,relu=self.layers[0]
        self.layers[0]=(weight*ball_matrix(embedding),bias,relu)
        self.readout_norm=sum((self.readout[i,j]**2
                              for i in range(self.readout.nrows())
                              for j in range(self.readout.ncols())),arb(0)).sqrt()
        if self.readout.nrows()!=2:
            raise ValueError('Circular witness needs exactly two readout outputs')

    def bound(self,code,point=False):
        i,j,depth=decode_box(code)
        theta,phi=angle_box(i,depth,point),angle_box(j,depth,point)
        angle=self.winding*phi
        x=arb_mat([[theta.cos()],[theta.sin()],[angle.cos()],[angle.sin()]])
        tangent=arb_mat([[0],[0],[-self.winding*angle.sin()],
                         [self.winding*angle.cos()]])
        masks=[]
        strict=True
        for weight,bias,relu in self.layers:
            before=weight*x+bias
            if not relu:
                masks.append([arb(1)]*before.nrows())
                x=before
                continue
            mask=[]
            x=arb_mat(before.nrows(),1)
            for row in range(before.nrows()):
                value=before[row,0]
                if value>0:
                    mask.append(arb(1));x[row,0]=value
                elif value<0:
                    mask.append(arb(0));x[row,0]=arb(0)
                else:
                    # Including either derivative at exact zero is conservative.
                    strict=False
                    mask.append(arb(.5,.5));x[row,0]=value.max(arb(0))
            masks.append(mask)
        backward=self.readout
        for (weight,_,_),mask in zip(reversed(self.layers),reversed(masks)):
            backward=arb_mat([[backward[row,col]*mask[col]
                               for col in range(backward.ncols())]
                              for row in range(2)])*weight
        derivative=backward*tangent
        residual=[derivative[0,0]+phi.sin(),derivative[1,0]-phi.cos()]
        upper=(residual[0].abs_upper()**2+residual[1].abs_upper()**2).sqrt().upper()
        lower=(residual[0]**2+residual[1]**2).sqrt().lower()
        return upper,lower,strict


def verify_witness(network,maximum_boxes=20000,maximum_depth=16,seconds=3300):
    started=time.monotonic()
    pending=[''];leaves=[];checked=0;maximum=arb(0)
    threshold=2/arb.pi()
    while pending:
        if checked>=maximum_boxes or time.monotonic()-started>=seconds:
            break
        code=pending.pop();checked+=1
        upper,_,_=network.bound(code)
        if upper.is_finite() and upper<threshold:
            leaves.append(code)
            maximum=maximum.max(upper).upper()
            continue
        _,point_lower,strict=network.bound(code,point=True)
        if strict and point_lower.is_finite() and point_lower>threshold:
            return dict(status='fixed_witness_rejected',continuous_certificate=False,
                        checked_boxes=checked,certified_leaves=len(leaves),
                        rejection_box=code,point_residual_lower=point_lower.str(50),
                        scope='This fixed derivative witness fails; fiber injectivity is not refuted')
        if len(code)>=maximum_depth:
            pending.append(code)
            break
        pending.extend(code+str(child) for child in [3,2,1,0])
    if pending:
        return dict(status='inconclusive',continuous_certificate=False,
                    checked_boxes=checked,certified_leaves=len(leaves),
                    unresolved_boxes=len(pending),maximum_certified_depth=max(map(len,leaves),default=None),
                    elapsed_seconds=time.monotonic()-started,
                    scope='Budget/depth/enclosure limit; no continuum injectivity claim')
    check_cover(leaves)
    # Serialize a containing ball, then re-check it before using it as epsilon.
    epsilon=arb(maximum.str(60))
    factor=(1-epsilon*arb.pi()/2)/network.readout_norm
    if not epsilon.contains(maximum) or not epsilon<threshold or not factor>0:
        return dict(status='inconclusive',continuous_certificate=False,
                    checked_boxes=checked,scope='Global separation factor is not rigorously positive')
    return dict(status='certified_fiber_injective',continuous_certificate=True,
                checked_boxes=checked,leaf_codes=leaves,
                epsilon_enclosure=epsilon.str(60),
                chord_lower_factor_enclosure=factor.str(60),
                elapsed_seconds=time.monotonic()-started,
                scope='Within every fixed-theta smooth nuisance circle; not full torus injectivity')


def replay_certificate(network,result):
    if result.get('status')!='certified_fiber_injective' or result.get('continuous_certificate') is not True:
        raise ValueError('Not a positive continuum certificate')
    leaves=result['leaf_codes']
    check_cover(leaves)
    maximum=arb(0)
    for code in leaves:
        upper,_,_=network.bound(code)
        if not upper.is_finite() or not upper<2/arb.pi():
            raise ValueError('Continuum certificate leaf fails interval replay: '+code)
        maximum=maximum.max(upper).upper()
    epsilon=arb(result['epsilon_enclosure'])
    if not epsilon.contains(maximum) or not epsilon<2/arb.pi():
        raise ValueError('Claimed global epsilon does not enclose replayed bounds')
    factor=(1-epsilon*arb.pi()/2)/network.readout_norm
    claimed_factor=arb(result['chord_lower_factor_enclosure'])
    if not factor>0 or not claimed_factor>0 or not claimed_factor.contains(factor):
        raise ValueError('Replayed global separation factor is not positive')
    return True
