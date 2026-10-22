import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.collect_production_noise import collect,validate_cell
from scripts.evaluate_production_noise import plan,source_files
from scripts.recover_release_evidence import digest,load_json

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/production_noise_completion_v1'


def sample(pilot=False):
    rows=plan(REPO)
    for row in rows:
        path=ROOT/(row['record']['run_id']+'.json')
        result=load_json(path.read_bytes())
        if not pilot or result['binding']['run_directory'].startswith('results/pilot/'):
            receipt=load_json(path.with_name(path.stem+'.replay.json').read_bytes())
            return row,result,receipt,digest(path)
    raise AssertionError('Expected retained shared primary/pilot source')


def validate(row,result,receipt,sha):
    protocol=load_json((REPO/'configs/production_noise_completion_v1.json').read_bytes())
    sources={p.relative_to(REPO).as_posix():digest(p) for p in source_files(REPO)}
    validate_cell(REPO,row,result,receipt,sha,sources,protocol)


def test_complete_real_noise_collection_and_replays_are_source_checked():
    result=collect(REPO,ROOT)
    assert result==load_json((ROOT/'collection.json').read_bytes())
    assert result['complete_unique_models']==305 and result['numeric_replays']==305
    assert result['main_models']==130 and result['relevance_models']==210 and result['shared_models']==35
    assert result['unreached_models']==0


def test_canonical_pilot_location_of_a_shared_primary_model_is_preserved():
    row,result,receipt,sha=sample(pilot=True)
    assert row['cohort']=='primary'
    validate(row,result,receipt,sha)


@pytest.mark.parametrize('damage,match',[
    ('checkpoint','checkpoint binding'),('scaling','curve/scaling'),('replay','replay receipt'),
    ('claim','source, protocol'),('escape','directory'),('level','Incomplete noise-level'),('roles','source, protocol')])
def test_changed_data_or_scientific_claim_rejected_even_with_updated_receipt(damage,match):
    row,result,receipt,sha=sample()
    result,receipt=copy.deepcopy(result),copy.deepcopy(receipt)
    if damage=='checkpoint':result['binding']['checkpoint_sha256']='0'*64
    elif damage=='scaling':result['noise_rows'][1]['coordinate_noise_sd']*=2
    elif damage=='replay':receipt['numeric_decoder_replay']=False
    elif damage=='claim':result['information_destruction_proved']=True
    elif damage=='escape':result['binding']['run_directory']='../outside/'+row['record']['run_id']
    elif damage=='level':result['noise_rows'].pop()
    else:result['binding']['roles']=['main','relevance']
    sha=hashlib.sha256(json.dumps(result).encode()).hexdigest()
    receipt['result_sha256']=sha;receipt['binding']=copy.deepcopy(result['binding'])
    with pytest.raises(ValueError,match=match):validate(row,result,receipt,sha)
