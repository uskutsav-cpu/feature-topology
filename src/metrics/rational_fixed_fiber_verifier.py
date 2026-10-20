"""Complete-circle certificates with conservative, exactly encoded proof bounds.

V3 printed a narrow Arb enclosure and required its decimal round trip to contain
a newly computed enclosure. That is stronger than the needed inequality and can
fail at the last printed digit. V4 uses exact dyadic upper/lower bounds instead;
the network enclosures, witnesses, and mathematical criterion are unchanged.
"""
import time

from flint import arb
from src.metrics.fixed_fiber_verifier import SCOPE, check_circle_cover

EPSILON_BITS = 20
FACTOR_BITS = 64


def dyadic_bound(point, bits, upward):
    """Round an exact finite Arb endpoint outwards, with one extra dyadic unit."""
    mantissa, exponent = map(int, point.man_exp())
    shift = exponent + bits
    if shift >= 0:
        integer = mantissa << shift
    else:
        denominator = 1 << -shift
        integer = -((-mantissa) // denominator) if upward else mantissa // denominator
    integer += 1 if upward else -1
    return {'numerator': str(integer), 'denominator': str(1 << bits)}


def exact_bound(encoded, bits):
    if (not isinstance(encoded, dict) or set(encoded) != {'numerator', 'denominator'}
            or encoded['denominator'] != str(1 << bits)
            or not isinstance(encoded['numerator'], str)):
        raise ValueError('Malformed exact dyadic proof bound')
    numerator = int(encoded['numerator'])
    if str(numerator) != encoded['numerator']:
        raise ValueError('Noncanonical proof numerator')
    value = arb(numerator) / (1 << bits)
    if not value.is_finite() or not value.is_exact():
        raise ValueError('Proof bound is not exactly representable at this precision')
    return value


def separation_bound(network, epsilon):
    if not network.readout_norm.is_finite() or not network.readout_norm > 0:
        raise ValueError('Readout norm has no finite positive enclosure')
    return (1 - epsilon * arb.pi() / 2) / network.readout_norm


def verify_circle(network, maximum_boxes=20000, maximum_depth=18, seconds=3300):
    started = time.monotonic()
    pending, leaves, checked, maximum = [''], [], 0, arb(0)
    threshold = 2 / arb.pi()
    # A small, stricter guard leaves room for rounding epsilon upwards.
    acceptance = threshold - arb(4) / (1 << EPSILON_BITS)
    common = dict(theta_fraction=network.theta_fraction, scope=SCOPE)
    while pending:
        if checked >= maximum_boxes or time.monotonic() - started >= seconds:
            break
        code = pending.pop()
        checked += 1
        upper, _, _ = network.bound(code)
        if upper.is_finite() and upper < acceptance:
            leaves.append(code)
            maximum = maximum.max(upper).upper()
            continue
        _, point_lower, strict = network.bound(code, point=True)
        if strict and point_lower.is_finite() and point_lower > threshold:
            return dict(**common, status='fixed_witness_rejected', continuous_certificate=False,
                        checked_boxes=checked, rejection_interval=code,
                        point_residual_lower=point_lower.str(60))
        if len(code) >= maximum_depth:
            pending.append(code)
            break
        pending.extend([code + '1', code + '0'])
    if pending:
        return dict(**common, status='inconclusive', continuous_certificate=False,
                    checked_boxes=checked, certified_leaves=len(leaves),
                    unresolved_intervals=len(pending), elapsed_seconds=time.monotonic() - started)
    check_circle_cover(leaves)
    epsilon_encoding = dyadic_bound(maximum, EPSILON_BITS, upward=True)
    epsilon = exact_bound(epsilon_encoding, EPSILON_BITS)
    factor = separation_bound(network, epsilon)
    factor_encoding = dyadic_bound(factor.lower(), FACTOR_BITS, upward=False)
    lower = exact_bound(factor_encoding, FACTOR_BITS)
    if not maximum <= epsilon or not epsilon < threshold or not lower > 0 or not lower <= factor:
        return dict(**common, status='inconclusive', continuous_certificate=False,
                    checked_boxes=checked, reason='No rigorously positive exact separation bound')
    return dict(**common, status='certified_fixed_nuisance_circle', continuous_certificate=True,
                checked_boxes=checked, leaf_codes=leaves,
                epsilon_upper_rational=epsilon_encoding, chord_lower_factor_rational=factor_encoding,
                elapsed_seconds=time.monotonic() - started)


def replay_circle(network, result):
    if (result.get('status') != 'certified_fixed_nuisance_circle'
            or result.get('continuous_certificate') is not True
            or result.get('theta_fraction') != network.theta_fraction or result.get('scope') != SCOPE):
        raise ValueError('Certificate status or task-coordinate scope changed')
    epsilon = exact_bound(result['epsilon_upper_rational'], EPSILON_BITS)
    claimed_lower = exact_bound(result['chord_lower_factor_rational'], FACTOR_BITS)
    if not epsilon >= 0 or not epsilon < 2 / arb.pi():
        raise ValueError('Exact epsilon does not meet the separation criterion')
    check_circle_cover(result['leaf_codes'])
    for code in result['leaf_codes']:
        upper, _, _ = network.bound(code)
        if not upper.is_finite() or not upper <= epsilon:
            raise ValueError('Circle interval exceeds the declared exact epsilon')
    factor = separation_bound(network, epsilon)
    if not claimed_lower > 0 or not claimed_lower <= factor:
        raise ValueError('Declared exact separation lower bound is not justified')
    return True
