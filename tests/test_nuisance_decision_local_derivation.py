import copy

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts.replay_nuisance_decision_collection_v2 import derived_claim
from src.metrics.nuisance_decision_verifier import replay_decision, verify_decision
from tests.test_nuisance_decision_verifier import classifier


def test_fixed_weaker_margin_is_exact_and_does_not_modify_original_or_domain():
    network = classifier()
    result = verify_decision(network)
    original = copy.deepcopy(result)
    derived = derived_claim(result, 'a'*64)
    assert result==original
    assert derived['leaf_codes']==original['leaf_codes']
    assert derived['theta_fraction']==original['theta_fraction']
    assert int(derived['scaled_margin_lower_rational']['numerator'])==int(original['scaled_margin_lower_rational']['numerator'])//2
    assert replay_decision(network, derived)
    result['scaled_margin_lower_rational']['numerator']='1'
    with pytest.raises(ValueError, match='not strictly positive'):
        derived_claim(result, 'a'*64)


def test_negative_point_claims_are_unchanged_and_replayed_as_negative():
    network = classifier(violation=True)
    result = verify_decision(network)
    derived = derived_claim(result, 'b'*64)
    assert derived['decision_certificate'] is False
    assert derived['counterexample_certificate'] is True
    assert derived['point_margin_upper_rational']==result['point_margin_upper_rational']
    assert derived['violation_code']==result['violation_code']
    assert replay_decision(network, derived)
