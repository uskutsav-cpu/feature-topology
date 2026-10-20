"""Arb affine-input/remainder enclosures for the same fixed fiber witnesses.

Unlike plain interval forward propagation, the four input coordinates remain
symbolic through affine layers. ReLU relaxation errors are conservatively kept
in a separate interval remainder. This reduces dependency overestimation; the
universal derivative criterion and conservative activation masks are unchanged.
The centered ReLU relaxation is standard (DeepZ, Singh et al., NeurIPS 2018);
this implementation does not claim a new abstract domain.
"""
from flint import arb,arb_mat
from src.metrics.continuous_fiber_verifier import CircularIntervalNetwork,angle_box,decode_box


class AffineCircularIntervalNetwork(CircularIntervalNetwork):
    def bound(self,code,point=False):
        i,j,depth=decode_box(code)
        theta,phi=angle_box(i,depth,point),angle_box(j,depth,point)
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
                masks.append([arb(1)]*before.nrows())
                continue
            new_center=arb_mat(before.nrows(),1)
            new_linear=arb_mat(before.nrows(),4)
            new_remainder=arb_mat(before.nrows(),1)
            mask=[]
            for row in range(before.nrows()):
                value=before[row,0]
                if value>=0:
                    strict=strict and bool(value>0)
                    mask.append(arb(1))
                    new_center[row,0]=center[row,0]
                    new_remainder[row,0]=remainder[row,0]
                    for col in range(4):new_linear[row,col]=linear[row,col]
                elif value<=0:
                    strict=strict and bool(value<0)
                    mask.append(arb(0))
                else:
                    strict=False;mask.append(arb(.5,.5))
                    lower,upper=value.lower(),value.upper()
                    # For y in [l,u], ReLU(y) lies in alpha*y+beta+[-beta,beta].
                    # Arb encloses every arithmetic operation, including alpha.
                    alpha=upper/(upper-lower)
                    beta=-alpha*lower/2
                    new_center[row,0]=alpha*center[row,0]+beta
                    new_remainder[row,0]=alpha*remainder[row,0]+arb(0,beta.abs_upper())
                    for col in range(4):new_linear[row,col]=alpha*linear[row,col]
            masks.append(mask)
            center,linear,remainder=new_center,new_linear,new_remainder
        backward=self.readout
        for (weight,_,_),mask in zip(reversed(self.layers),reversed(masks)):
            backward=arb_mat([[backward[row,col]*mask[col]
                               for col in range(backward.ncols())]
                              for row in range(2)])*weight
        tangent=arb_mat([[0],[0],[-self.winding*angle.sin()],[self.winding*angle.cos()]])
        derivative=backward*tangent
        residual=[derivative[0,0]+phi.sin(),derivative[1,0]-phi.cos()]
        upper=(residual[0].abs_upper()**2+residual[1].abs_upper()**2).sqrt().upper()
        lower=(residual[0]**2+residual[1]**2).sqrt().lower()
        return upper,lower,strict
