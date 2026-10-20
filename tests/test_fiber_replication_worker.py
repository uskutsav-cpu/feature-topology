import copy
import json
from pathlib import Path
import shutil

import pytest

pytest.importorskip('flint')
pytest.importorskip('torch')
from scripts.remote_fiber_replication import buffered_decision
from scripts.screen_fiber_replication import source_files
from scripts.validate_fiber_replication_screens import validate
from src.metrics.nuisance_decision_verifier import replay_decision, verify_decision
from tests.test_nuisance_decision_verifier import classifier


def screen_fixture(tmp_path):
    source=Path(__file__).resolve().parents[1]
    for path in [*source_files(source),source/'configs/completion/primary_inputs_v3.json']:
        target=tmp_path/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    root=tmp_path/'results/fiber_replication_screens_v1'
    shutil.copytree(source/'results/fiber_replication_screens_v1',root)
    return root


def test_prospective_buffer_changes_only_positive_margin_and_keeps_it_replayable():
    network=classifier();result=verify_decision(network);original=copy.deepcopy(result)
    buffered=buffered_decision(result)
    assert result==original
    assert buffered['leaf_codes']==original['leaf_codes']
    assert int(buffered['scaled_margin_lower_rational']['numerator'])==int(original['scaled_margin_lower_rational']['numerator'])//2
    assert replay_decision(network,buffered)
    result['scaled_margin_lower_rational']['numerator']='1'
    failed=buffered_decision(result)
    assert failed['status']=='inconclusive' and failed['decision_certificate'] is False
    assert 'scaled_margin_lower_rational' not in failed and 'leaf_codes' not in failed


def test_negative_point_certificate_keeps_its_original_bound_and_status():
    network=classifier(violation=True);result=verify_decision(network)
    assert buffered_decision(result)==result
    assert replay_decision(network,buffered_decision(result))


def test_complete_screen_grid_keeps_failed_witnesses_visible(tmp_path):
    root=screen_fixture(tmp_path)
    collection=validate(tmp_path,root)
    assert collection['complete_models']==36 and collection['planned_domains']==144
    assert collection['eligible_models']+collection['screen_failed_models']==36
    assert {r['seed'] for r in collection['records']}==set(range(1,10))


@pytest.mark.parametrize('damage',['screen','readout','missing'])
def test_coherent_json_edits_cannot_change_screen_eligibility_or_drop_models(tmp_path,damage):
    root=screen_fixture(tmp_path)
    collection=json.loads((root/'collection.json').read_text())
    row=next(row for row in collection['records'] if not row['empirical_screen_passed'])
    if damage=='screen':
        row['empirical_screen_passed']=True;row['status']='eligible_fixed_witness'
        collection['eligible_models']+=1;collection['screen_failed_models']-=1
    elif damage=='readout':row['readout_hex'][0].pop()
    else:collection['records'].pop()
    (root/(row['run_id']+'.json')).write_text(json.dumps(row))
    (root/'collection.json').write_text(json.dumps(collection))
    with pytest.raises((ValueError,TypeError),match='(changed|Incomplete|inhomogeneous)'):
        validate(tmp_path,root)
