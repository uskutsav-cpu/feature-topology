import json
from pathlib import Path

from scripts.finalize_cifar_hosted_20261021 import component_cells
from scripts.remote_cifar import cell_id
from scripts.run_cifar_remote_queue import production_cells


def test_hosted_finalization_reuses_exact_published_original_cohorts_and_frozen_missing_cells():
    repo=Path(__file__).resolve().parents[1]
    for dataset in ['CIFAR10','CIFAR100']:
        key=dataset.lower();frozen=json.loads((repo/'results'/key/'gamma_to_lr.json').read_text())
        original,correction,phase=component_cells(dataset,production_cells(dataset,frozen))
        published=json.loads((Path(__file__).parent/'fixtures/cifar'/
            (key+'_production_original_complete_component.json')).read_text())
        assert {cell_id(c) for c in original}==set(published['complete'])
        assert {c['seed'] for c in phase}==({0,2,4} if dataset=='CIFAR10' else set(range(5)))
        assert all(c['gamma']==.125 for c in phase)
        if dataset=='CIFAR10':
            assert len(correction)==5 and all(c['gamma']==128. for c in correction)
            assert {c['lr'].hex() for c in correction}=={'0x1.21a1851ff630bp+2'}
        else:assert correction==[]


def test_transport_retry_never_hides_unfinished_training_or_corruption():
    from scripts.finalize_cifar_hosted_20261021 import transient_collection_failure
    assert transient_collection_failure({'invalid':[{'error':'read tcp: connection reset by peer'}]})
    assert not transient_collection_failure({'invalid':[], 'missing':['unfinished-cell']})
    assert not transient_collection_failure({'invalid':[{'error':'Hosted CIFAR archive checksum mismatch'}]})
    assert not transient_collection_failure({'invalid':[
        {'error':'connection reset by peer'}, {'error':'Hosted CIFAR source identity changed'}]})
