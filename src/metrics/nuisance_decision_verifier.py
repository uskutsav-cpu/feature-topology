"""Arb bounds for correct class decisions of the complete centered MLP."""
import time

from flint import arb, arb_mat
from src.metrics.continuous_fiber_verifier import angle_box, ball_matrix
from src.metrics.fixed_fiber_verifier import check_circle_cover
from src.metrics.rational_fixed_fiber_verifier import dyadic_bound, exact_bound

MARGIN_BITS = 64
SCOPE = ('Correct top-one class on the specified fixed-theta nuisance circle in '
         'the complete real-valued centered classifier; no full-torus or machine-arithmetic claim')


class CenteredCircleClassifier:
    def __init__(self, current_layers, reference_layers, embedding, gamma, theta_numerator):
        if theta_numerator not in {1,3,5,7}:
            raise ValueError('Expected a declared class-sector midpoint')
        self.theta_fraction = [theta_numerator,8]
        self.target_class = theta_numerator//2
        self.gamma = arb(float(gamma))
        if not self.gamma.is_finite() or not self.gamma>0:
            raise ValueError('Classifier scaling must be finite and positive')
        self.current = self.layers(current_layers, embedding)
        self.reference = self.layers(reference_layers, embedding)

    @staticmethod
    def layers(values, embedding):
        layers = [(ball_matrix(weight),arb_mat([[arb(float(v))] for v in bias]),relu)
                  for weight,bias,relu in values]
        if not layers or layers[-1][0].nrows()!=4 or layers[-1][2]:
            raise ValueError('Expected a complete network with four linear output logits')
        weight,bias,relu = layers[0]
        layers[0] = (weight*ball_matrix(embedding),bias,relu)
        if layers[0][0].ncols()!=4:
            raise ValueError('Expected four latent circular input coordinates')
        return layers

    @staticmethod
    def forms(layers, input_center, displacement):
        center = input_center
        linear = arb_mat([[int(row==col) for col in range(4)] for row in range(4)])
        remainder = arb_mat(4,1)
        for weight,bias,relu in layers:
            center,linear,remainder = weight*center+bias,weight*linear,weight*remainder
            if not relu:
                continue
            before = center+linear*displacement+remainder
            next_center = arb_mat(before.nrows(),1)
            next_linear = arb_mat(before.nrows(),4)
            next_remainder = arb_mat(before.nrows(),1)
            for row in range(before.nrows()):
                value = before[row,0]
                if value>=0:
                    next_center[row,0] = center[row,0]
                    next_remainder[row,0] = remainder[row,0]
                    for col in range(4):next_linear[row,col] = linear[row,col]
                elif value<=0:
                    continue
                else:
                    lower,upper = value.lower(),value.upper()
                    alpha = upper/(upper-lower)
                    beta = -alpha*lower/2
                    next_center[row,0] = alpha*center[row,0]+beta
                    next_remainder[row,0] = alpha*remainder[row,0]+arb(0,beta.abs_upper())
                    for col in range(4):next_linear[row,col] = alpha*linear[row,col]
            center,linear,remainder = next_center,next_linear,next_remainder
        return center,linear,remainder

    def margins(self, code, point=False):
        if any(char not in '01' for char in code):
            raise ValueError('Invalid binary circle interval')
        phi = angle_box(int(code,2) if code else 0,len(code),point)
        theta = 2*arb.pi()*self.theta_fraction[0]/8
        coordinates = arb_mat([[theta.cos()],[theta.sin()],[phi.cos()],[phi.sin()]])
        center = arb_mat([[coordinates[row,0].mid()] for row in range(4)])
        displacement = coordinates-center
        current = self.forms(self.current,center,displacement)
        reference = self.forms(self.reference,center,displacement)
        y = self.target_class
        margins = {}
        for competitor in range(4):
            if competitor==y:continue
            c = current[0][y,0]-current[0][competitor,0]-reference[0][y,0]+reference[0][competitor,0]
            r = current[2][y,0]-current[2][competitor,0]-reference[2][y,0]+reference[2][competitor,0]
            value = c+r
            for col in range(4):
                coefficient = (current[1][y,col]-current[1][competitor,col]
                               -reference[1][y,col]+reference[1][competitor,col])
                value += coefficient*displacement[col,0]
            margins[competitor] = value/self.gamma
        return margins


def verify_decision(network, maximum_boxes=20000, maximum_depth=18, seconds=3300):
    started = time.monotonic()
    pending,leaves,checked,minimum = [''],[],0,None
    common = dict(theta_fraction=network.theta_fraction,target_class=network.target_class,scope=SCOPE)
    while pending:
        if checked>=maximum_boxes or time.monotonic()-started>=seconds:break
        code = pending.pop();checked += 1
        margins = network.margins(code)
        if all(value.is_finite() and value>0 for value in margins.values()):
            leaves.append(code)
            for value in margins.values():
                lower = value.lower()
                minimum = lower if minimum is None else minimum.min(lower).lower()
            continue
        for competitor,value in network.margins(code,point=True).items():
            if value.is_finite() and value<0:
                encoding = dyadic_bound(value.upper(),MARGIN_BITS,upward=True)
                upper = exact_bound(encoding,MARGIN_BITS)
                if value<=upper and upper<0:
                    return dict(**common,status='target_class_violation',decision_certificate=False,
                        counterexample_certificate=True,checked_boxes=checked,violation_code=code,
                        violating_class=competitor,point_margin_upper_rational=encoding)
        if len(code)>=maximum_depth:
            pending.append(code);break
        pending.extend([code+'1',code+'0'])
    if pending:
        return dict(**common,status='inconclusive',decision_certificate=False,
            counterexample_certificate=False,checked_boxes=checked,certified_leaves=len(leaves),
            unresolved_intervals=len(pending),elapsed_seconds=time.monotonic()-started)
    check_circle_cover(leaves)
    encoding = dyadic_bound(minimum,MARGIN_BITS,upward=False)
    lower = exact_bound(encoding,MARGIN_BITS)
    if not lower>0 or not lower<=minimum:
        return dict(**common,status='inconclusive',decision_certificate=False,
            counterexample_certificate=False,checked_boxes=checked,
            reason='No positive exact dyadic margin')
    return dict(**common,status='certified_correct_class_circle',decision_certificate=True,
        counterexample_certificate=False,checked_boxes=checked,leaf_codes=leaves,
        scaled_margin_lower_rational=encoding,elapsed_seconds=time.monotonic()-started)


def replay_decision(network, result):
    if (result.get('theta_fraction')!=network.theta_fraction
            or result.get('target_class')!=network.target_class or result.get('scope')!=SCOPE):
        raise ValueError('Classifier decision domain, target or scope changed')
    if (result.get('status')=='target_class_violation' and result.get('decision_certificate') is False
            and result.get('counterexample_certificate') is True):
        competitor = result['violating_class']
        margins = network.margins(result['violation_code'],point=True)
        upper = exact_bound(result['point_margin_upper_rational'],MARGIN_BITS)
        if (competitor not in margins or not upper<0 or not margins[competitor].is_finite()
                or not margins[competitor]<=upper):
            raise ValueError('Target-class counterexample failed replay')
        return True
    if (result.get('status')!='certified_correct_class_circle'
            or result.get('decision_certificate') is not True
            or result.get('counterexample_certificate') is not False):
        raise ValueError('Not a certified classifier decision claim')
    check_circle_cover(result['leaf_codes'])
    lower = exact_bound(result['scaled_margin_lower_rational'],MARGIN_BITS)
    if not lower>0:
        raise ValueError('Declared classifier margin is not strictly positive')
    for code in result['leaf_codes']:
        if any(not value.is_finite() or not lower<=value for value in network.margins(code).values()):
            raise ValueError('Declared classifier margin exceeds a replayed leaf bound')
    return True
