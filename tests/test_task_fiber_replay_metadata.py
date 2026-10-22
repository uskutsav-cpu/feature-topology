import json
import pytest
import torch

from scripts.evaluate_production_noise import close_numeric
from scripts.replay_task_preserving_fibers_v2 import compare_result


def test_torch_version_json_type_roundtrip_is_metadata_not_a_mathematical_difference():
    actual={'score':.125,'status':'evaluated'};host={'torch':torch.__version__,'threads':2}
    stored=json.loads(json.dumps(dict(**actual,host=host,source_commit='a'*40)))
    with pytest.raises(ValueError,match='value changed'):
        close_numeric(dict(**actual,host=host,source_commit='a'*40),stored)
    compare_result(actual,stored,'a'*40,dict(torch=str(torch.__version__),threads=2))


@pytest.mark.parametrize('change',['numeric','version','source','threads_type'])
def test_metadata_replay_fix_cannot_accept_changed_math_source_or_runtime(change):
    actual={'score':.125,'status':'evaluated'};host={'torch':'2.14.0','threads':2}
    stored=dict(**actual,host=dict(host),source_commit='a'*40)
    if change=='numeric':stored['score']=.126
    elif change=='version':stored['host']['torch']='other'
    elif change=='source':stored['source_commit']='b'*40
    else:stored['host']['threads']=2.
    with pytest.raises(ValueError):compare_result(actual,stored,'a'*40,host)
