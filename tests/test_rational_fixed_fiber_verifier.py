import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('flint')
from flint import arb, ctx
from src.metrics.rational_fixed_fiber_verifier import (
    EPSILON_BITS, FACTOR_BITS, dyadic_bound, exact_bound, replay_circle, verify_circle,
)
from tests.test_joint_fiber_verifier import coupled_circle
from tests.test_fixed_fiber_verifier import network


def test_exact_bounds_survive_json_and_a_separate_replay_process(tmp_path):
    net = coupled_circle()
    result = verify_circle(net, maximum_boxes=2000, seconds=60)
    assert result['continuous_certificate'] is True
    path = tmp_path / 'certificate.json'
    path.write_text(json.dumps(result))
    assert replay_circle(net, json.loads(path.read_text()))
    subprocess.run([sys.executable, '-c',
        'import json,sys; from tests.test_joint_fiber_verifier import coupled_circle; '
        'from src.metrics.rational_fixed_fiber_verifier import replay_circle; '
        'assert replay_circle(coupled_circle(),json.load(open(sys.argv[1])))', str(path)],
        cwd=Path(__file__).resolve().parents[1], check=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])))


def test_replay_rejects_understated_error_and_overstated_separation_or_scope():
    net = coupled_circle()
    result = verify_circle(net, maximum_boxes=2000, seconds=60)
    damaged = copy.deepcopy(result)
    damaged['epsilon_upper_rational']['numerator'] = '0'
    with pytest.raises(ValueError, match='exceeds'): replay_circle(net, damaged)
    damaged = copy.deepcopy(result)
    damaged['chord_lower_factor_rational']['numerator'] = str(10 << FACTOR_BITS)
    with pytest.raises(ValueError, match='not justified'): replay_circle(net, damaged)
    damaged = copy.deepcopy(result)
    damaged['leaf_codes'].pop()
    with pytest.raises(ValueError, match='unresolved'): replay_circle(net, damaged)
    damaged = copy.deepcopy(result)
    damaged['theta_fraction'] = [3, 8]
    with pytest.raises(ValueError, match='scope changed'): replay_circle(net, damaged)


@pytest.mark.parametrize('projection,winding', [(True, 1), (False, 2)])
def test_information_losing_controls_remain_uncertified(projection, winding):
    result = verify_circle(network(projection, winding), maximum_boxes=2000, seconds=60)
    assert result['continuous_certificate'] is False
    assert result['status'] == 'fixed_witness_rejected'


def test_outward_integer_rounding_of_both_signed_endpoints():
    ctx.prec = 128
    for value in [arb(0), arb(1) / 3, -arb(1) / 3, arb(7) / 8]:
        upper = exact_bound(dyadic_bound(value.upper(), EPSILON_BITS, True), EPSILON_BITS)
        lower = exact_bound(dyadic_bound(value.lower(), FACTOR_BITS, False), FACTOR_BITS)
        assert upper > value and lower < value
    with pytest.raises(ValueError, match='Malformed'):
        exact_bound({'numerator': '1', 'denominator': '3'}, EPSILON_BITS)
