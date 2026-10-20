import copy

import pytest

torch = pytest.importorskip('torch')
from scripts.remote_checkpoint_equivalence import compare_states


def test_alias_check_includes_frozen_reference_and_rejects_one_changed_bit():
    selected = dict(config={'seed': 0}, step=32,
                    model={'network.weight': torch.tensor([[1., 2.]]),
                           'initial.weight': torch.tensor([[.5, .25]])})
    final = copy.deepcopy(selected)
    final['metadata'] = 'Different serialization metadata is allowed'
    assert len(compare_states(selected, final, {'seed': 0}, 32)) == 64
    final['model']['initial.weight'][0, 0] = torch.nextafter(torch.tensor(.5), torch.tensor(1.))
    with pytest.raises(ValueError, match='initial.weight'):
        compare_states(selected, final, {'seed': 0}, 32)


def test_alias_check_rejects_same_tensors_at_another_step():
    selected = dict(config={}, step=8, model={'network.weight': torch.ones(2)})
    final = copy.deepcopy(selected)
    final['step'] = 9
    with pytest.raises(ValueError, match='task/step'):
        compare_states(selected, final, {}, 8)
