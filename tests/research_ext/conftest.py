from pathlib import Path
from copy import deepcopy
import pytest
from research_ext.io import atomic_json,original_fingerprint

@pytest.fixture
def make_run(tmp_path):
    def make(gamma=.5,seed=0,root=None,extra=None):
        root=Path(root or tmp_path/'study')
        c=dict(gamma=gamma,seed=seed,width=4,depth=1,target_loss=.1,lr=.01)
        c.update(extra or {})
        rid=original_fingerprint(c)
        run=root/'runs'/rid
        run.mkdir(parents=True)
        atomic_json(run/'config.json',c)
        atomic_json(run/'summary.json',dict(run_id=rid,config=c,status='converged',history=[
            dict(step=0,training_loss=1.4,training_accuracy=.25),
            dict(step=10,training_loss=.08,training_accuracy=.99)]))
        (run/'final.pt').touch(); (run/'step_0000000.pt').touch(); (run/'step_0000010.pt').touch()
        options=dict(all_checkpoints=True,jacobian_points=10,ntk_points=4,ph_size=8,
                     ph_repeats=1,ph_maxdim=1,probe_train=8,probe_test=8,probe_iterations=5)
        metrics=run/'metrics'/'test_profile'
        atomic_json(metrics/'options.json',options)
        row=dict(step=10,gamma=gamma,seed=seed,test_accuracy=.98,ntk_drift=.4,layers=[dict(
            layer=1,cka_drift=.2,effective_rank=2.,scale=1.,
            jacobian=dict(q01=.2,normalized_q01=.2,task_norm=1.,nuisance_norm=.9),
            global_margin=dict(q01=.1,normalized_q01=.1),collisions=dict(mean=.2),
            probes=dict(linear=dict(angular_cosine=.99,task_accuracy=.97),mlp=dict(angular_cosine=.998,task_accuracy=.98)),
            persistence=[dict(H1=dict(top1=.5,top2=.3))])])
        atomic_json(metrics/'final.json',row)
        atomic_json(metrics/'step_0000010.json',row)
        initial=deepcopy(row); initial.update(step=0,ntk_drift=0.)
        initial['layers'][0]['cka_drift']=0.
        atomic_json(metrics/'step_0000000.json',initial)
        return root,run,metrics
    return make
