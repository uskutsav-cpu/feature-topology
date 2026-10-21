import copy
from pathlib import Path

import pytest

from scripts.collect_hosted_task_fibers import validate_execution
from scripts.collect_task_preserving_fibers import collect,validate_cell
from scripts.evaluate_task_preserving_fibers import source_files,task_plan
from scripts.recover_release_evidence import digest,load_json

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/task_preserving_fiber_completion_hosted_v1'


def example(relevance):
    rows,protocol=task_plan(REPO)
    row=next(r for r in rows if r['relevance']==relevance)
    path=ROOT/(row['record']['run_id']+'.json')
    result=load_json(path.read_bytes())
    receipt=load_json((ROOT/(row['record']['run_id']+'.replay.json')).read_bytes())
    sources={p.relative_to(REPO).as_posix():digest(p) for p in source_files(REPO)}
    return row,result,receipt,digest(path),sources,protocol


def test_real_task_fiber_collection_and_every_execution_receipt_remain_source_bound():
    value=collect(REPO,ROOT)
    assert value==load_json((ROOT/'collection.json').read_bytes())
    assert value['complete_models']==value['numeric_replays']==210
    assert value['paired_family_views']==420 and value['distinct_family_evaluations']==385
    assert value['lambda_zero_family_reuse']==35 and value['unreached_models']==0
    audit=validate_execution(REPO,ROOT,'39535434835','7df70ce8422964953cfedebe18d44f916a6bfe40')
    assert audit==load_json((ROOT/'execution/audit.json').read_bytes())
    assert len(audit['input_sha256'])==638


@pytest.mark.parametrize('damage',['continuum','tangent_scale','search_domain','risk','family_count','budget'])
def test_task_fiber_semantic_checks_reject_changed_results_even_with_the_original_receipt(damage):
    row,result,receipt,sha,sources,protocol=example(.05)
    result=copy.deepcopy(result)
    if damage=='continuum':result['continuum_certificate']=True
    elif damage=='tangent_scale':result['families'][0]['local_per_unit_flat_torus_tangent']['normalized_minimum']*=2
    elif damage=='search_domain':result['families'][0]['adversarial_search']['best_candidate']['latent_pair'][0][0]+=.01
    elif damage=='risk':result['binding']['actual_training_loss']=.11
    elif damage=='family_count':result['families'].pop()
    else:result['families'][0]['adversarial_search']['evaluated_candidates']=1041
    with pytest.raises(ValueError):validate_cell(REPO,row,result,receipt,sha,sources,protocol)


def test_lambda_zero_control_cannot_be_silently_replaced_by_a_different_computation():
    row,result,receipt,sha,sources,protocol=example(0.)
    result['families'][1]['classifier_grid_accuracy']-=.0001
    with pytest.raises(ValueError,match='Lambda-zero'):
        validate_cell(REPO,row,result,receipt,sha,sources,protocol)
