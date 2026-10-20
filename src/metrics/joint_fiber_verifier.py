"""Preserve the joint readout direction by enumerating small activation sets.

For fixed angles, the derivative vector is multiaffine in the ReLU derivative
masks. It lies in the convex hull of binary mask vertices. The Euclidean norm
is therefore bounded by the largest vertex norm. Enumerating a few uncertain
masks avoids the larger componentwise rectangle; larger sets use a conservative
interval fallback. This is an established vertex/branching principle, not a
claim to solve general network verification efficiently.
"""
from itertools import product

from flint import arb,arb_mat
from src.metrics.fixed_fiber_verifier import FixedThetaIntervalNetwork


class JointMaskFixedThetaNetwork(FixedThetaIntervalNetwork):
    def __init__(self,*args,maximum_uncertain=6,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(maximum_uncertain,int) or not 0<=maximum_uncertain<=12:
            raise ValueError('Invalid activation enumeration limit')
        self.maximum_uncertain=maximum_uncertain

    def masks(self,theta,phi):
        angle=self.winding*phi
        coordinates=arb_mat([[theta.cos()],[theta.sin()],[angle.cos()],[angle.sin()]])
        center=arb_mat([[coordinates[row,0].mid()] for row in range(4)])
        displacement=coordinates-center
        linear=arb_mat([[int(row==col) for col in range(4)] for row in range(4)])
        remainder=arb_mat(4,1)
        masks=[];strict=True
        for weight,bias,relu in self.layers:
            center=weight*center+bias
            linear=weight*linear
            remainder=weight*remainder
            before=center+linear*displacement+remainder
            if not relu:
                masks.append([1]*before.nrows());continue
            new_center=arb_mat(before.nrows(),1)
            new_linear=arb_mat(before.nrows(),4)
            new_remainder=arb_mat(before.nrows(),1)
            mask=[]
            for row in range(before.nrows()):
                value=before[row,0]
                if value.is_zero():
                    # An exactly zero range is constant on a nondegenerate
                    # verification interval, so its phi derivative is zero.
                    # At a point, strict=False still prevents witness rejection.
                    strict=False;mask.append(0)
                elif value>=0:
                    strict=strict and bool(value>0);mask.append(1)
                    new_center[row,0]=center[row,0]
                    new_remainder[row,0]=remainder[row,0]
                    for col in range(4):new_linear[row,col]=linear[row,col]
                elif value<=0:
                    strict=strict and bool(value<0);mask.append(0)
                else:
                    strict=False;mask.append(None)
                    lower,upper=value.lower(),value.upper()
                    alpha=upper/(upper-lower);beta=-alpha*lower/2
                    new_center[row,0]=alpha*center[row,0]+beta
                    new_remainder[row,0]=alpha*remainder[row,0]+arb(0,beta.abs_upper())
                    for col in range(4):new_linear[row,col]=alpha*linear[row,col]
            masks.append(mask)
            center,linear,remainder=new_center,new_linear,new_remainder
        return masks,strict

    def bound(self,code,point=False):
        theta,phi=self.parameters(code,point)
        masks,strict=self.masks(theta,phi)
        uncertain=[(layer,col) for layer,row in enumerate(masks)
                   for col,value in enumerate(row) if value is None]
        angle=self.winding*phi
        tangent=arb_mat([[0],[0],[-self.winding*angle.sin()],[self.winding*angle.cos()]])

        def evaluate(assignment):
            backward=self.readout
            for layer in reversed(range(len(self.layers))):
                weight=self.layers[layer][0]
                derivative_masks=[]
                for col,value in enumerate(masks[layer]):
                    selected=assignment.get((layer,col),value)
                    derivative_masks.append(arb(.5,.5) if selected is None else arb(selected))
                backward=arb_mat([[backward[row,col]*derivative_masks[col]
                                   for col in range(backward.ncols())]
                                  for row in range(2)])*weight
            derivative=backward*tangent
            residual=[derivative[0,0]+phi.sin(),derivative[1,0]-phi.cos()]
            upper=(residual[0].abs_upper()**2+residual[1].abs_upper()**2).sqrt().upper()
            lower=(residual[0]**2+residual[1]**2).sqrt().lower()
            return upper,lower

        if not uncertain:
            upper,lower=evaluate({})
            return upper,lower if strict else arb(0),strict
        if len(uncertain)>self.maximum_uncertain:
            upper,_=evaluate({})
            return upper,arb(0),False
        maximum=arb(0)
        for bits in product([0,1],repeat=len(uncertain)):
            upper,_=evaluate(dict(zip(uncertain,bits)))
            maximum=maximum.max(upper).upper()
        return maximum,arb(0),False
