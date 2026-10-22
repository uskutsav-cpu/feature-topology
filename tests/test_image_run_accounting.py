import json

import pandas as pd
import pytest

from scripts.image_statistics import image_rows, image_run_accounting
from scripts.v3_results_document import image_accounting_table


def write_run(repo, rid, seed, status, loss=None, accuracy=.34):
    directory=repo/'results/dsprites/runs'/rid
    directory.mkdir(parents=True)
    config=dict(gamma=1.,seed=seed,target_loss=.2)
    history=[] if loss is None else [dict(step=5000,training_loss=loss)]
    (directory/'summary.json').write_text(json.dumps(dict(config=config,status=status,history=history)))
    if status!='diverged':
        (directory/'metrics.json').write_text(json.dumps(dict(test={'accuracy':accuracy,'loss':1.1},layers=[])))
    return directory


def test_divergent_outcomes_remain_counted_without_imputed_scores_and_budget_runs_keep_metrics(tmp_path):
    write_run(tmp_path,'converged',0,'converged',.1,.8)
    write_run(tmp_path,'exhausted',1,'budget_exhausted',.9,.34)
    write_run(tmp_path,'diverged',2,'diverged')
    frame=image_run_accounting(tmp_path).set_index('run_id')
    assert len(frame)==3
    assert frame.loc['exhausted','last_recorded_loss_le_target']==False
    assert frame.loc['exhausted','finite_metrics_included']==True
    assert frame.loc['exhausted','test_accuracy']==.34
    assert pd.isna(frame.loc['diverged','test_accuracy'])
    assert pd.isna(frame.loc['diverged','last_recorded_training_loss'])
    assert frame.loc['converged','target_loss']==.2
    assert set(frame.uniform_random_accuracy)=={1/3}
    metrics,_=image_rows(tmp_path)
    assert set(metrics.run_id)=={'converged','exhausted'}
    table=image_accounting_table(frame.reset_index())
    assert '| dsprites | 3 | 1 | 1 | 1 | 0.57 to 0.57 |' in table


def test_duplicate_seed_and_false_convergence_are_rejected(tmp_path):
    write_run(tmp_path,'first',0,'converged',.1)
    duplicate=write_run(tmp_path,'duplicate',0,'budget_exhausted',.9)
    with pytest.raises(ValueError,match='duplicate'):
        image_run_accounting(tmp_path)
    summary=json.loads((duplicate/'summary.json').read_text())
    summary['config']['seed']=1;summary['status']='converged'
    (duplicate/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='contradicts'):
        image_run_accounting(tmp_path)


def test_last_finite_log_from_a_divergent_run_is_not_a_test_outcome(tmp_path):
    write_run(tmp_path,'diverged',0,'diverged',.6)
    frame=image_run_accounting(tmp_path)
    assert frame.iloc[0].last_recorded_training_loss==.6
    assert pd.isna(frame.iloc[0].test_loss)
    assert 'not estimable' in image_accounting_table(frame)


def test_unknown_running_status_cannot_appear_in_final_accounting(tmp_path):
    write_run(tmp_path,'unfinished',0,'running',.9)
    with pytest.raises(ValueError,match='Unfinished or unknown'):
        image_run_accounting(tmp_path)
