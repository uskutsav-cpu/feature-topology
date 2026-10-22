import copy
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')
from scripts.screen_fiber_replication import replication_records


def inputs():
    repo=Path(__file__).resolve().parents[1]
    return (json.loads((repo/'configs/completion/primary_inputs_v3.json').read_text()),
            json.loads((repo/'configs/fiber_replication_v1.json').read_text()))


def test_replication_uses_every_remaining_seed_and_no_discovery_seed():
    index,protocol=inputs()
    rows=replication_records(index,protocol)
    assert len(rows)==36 and {row['seed'] for row in rows}==set(range(1,10))
    assert all(sum(row['gamma']==gamma for row in rows)==9 for gamma in [.03125,.5,1.,8.])


def test_missing_duplicate_or_changed_primary_source_cannot_be_dropped(tmp_path):
    index,protocol=inputs()
    rows=replication_records(index,protocol)
    index['runs']=[row for row in index['runs'] if row['run_id']!=rows[0]['run_id']]
    with pytest.raises(ValueError,match='Missing or duplicate'):
        replication_records(index,protocol)
    index['runs'].extend([rows[0],rows[0]])
    with pytest.raises(ValueError,match='Missing or duplicate'):
        replication_records(index,protocol)
    index['runs'].pop()
    rows[0]['config']['relevance']=.5
    with pytest.raises(ValueError,match='unchanged primary task'):
        replication_records(index,protocol)


def test_seed_exclusion_or_gamma_reselection_changes_the_recorded_study():
    index,protocol=inputs()
    for key,value in [('training_seeds',list(range(1,9))),('gammas',[.03125,.5,1.,16.])]:
        changed=copy.deepcopy(protocol);changed[key]=value
        with pytest.raises(ValueError,match='Different recorded'):
            replication_records(index,changed)
