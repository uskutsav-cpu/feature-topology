import pytest

from scripts.reconstruct_cifar_provenance import job_descriptor,resolve_job
from scripts.remote_cifar import cell_id,parse_cell


def test_decimal_matrix_display_cannot_invent_exact_learning_rate_provenance():
    with pytest.raises(ValueError,match='unique typed'):
        resolve_job('cell (production, CIFAR10, 128, 1, 4.52548339959391)',[])


def test_rounded_display_uses_typed_status_identity_and_preserves_ambiguity():
    exact=parse_cell(dict(stage='production',dataset='CIFAR10',gamma=128.,seed=1,lr_hex='0x1.21a1851ff630bp+2'))
    name='cell (production, CIFAR10, 128, 1, 4.52548339959391)'
    assert resolve_job(name,[exact])==exact
    other={**exact,'lr':4.52548339959391}
    assert cell_id(other)!=cell_id(exact)
    with pytest.raises(ValueError,match='unique typed'):resolve_job(name,[exact,other])


def test_exact_hex_worker_identity_does_not_roundtrip_through_decimal_display():
    cell=resolve_job('cell (production, CIFAR10, 128, 1, 0x1.21a1851ff630bp+2)',[])
    assert cell['lr'].hex()=='0x1.21a1851ff630bp+2'


def test_calibration_and_nonworker_jobs_are_distinguished():
    assert resolve_job('cell (calibration, CIFAR100, 0.125, 0.25)',[])==dict(
        stage='calibration',dataset='CIFAR100',gamma=.125,multiplier=.25)
    assert resolve_job('plan',[]) is None
    with pytest.raises(ValueError,match='Unrecognized'):job_descriptor('cell (production, CIFAR10, 128, 1)')
