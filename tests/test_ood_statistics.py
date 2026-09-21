import json
import pandas as pd

from scripts.ood_statistics import summarize_nuisance_shift


def test_nuisance_shift_statistics_preserve_seed_pairing(tmp_path):
    runs=[]
    for gamma in [.5,1.]:
        for seed in [0,1]:
            environments={}
            for name,offset in [('iid',0.),('concentrated',-.1),('spurious',-.4),('unseen',-.2)]:
                environments[name]={'accuracy':.8+gamma/10+seed/100+offset,
                                    'loss':.3-gamma/100-seed/100-offset,'samples':5000}
            runs.append(dict(run_id=f'{int(gamma*10)}{seed:015d}',status='evaluated',
                             training_nuisance_condition='iid',gamma=gamma,seed=seed,
                             selected_step=4,actual_training_loss=.08,
                             checkpoint_sha256=str(seed)*64,environments=environments))
    manifest={'schema':'feature-topology.nuisance-shift-manifest.v1',
              'evaluation_environments':['iid','concentrated','spurious','unseen'],'runs':runs}
    source=tmp_path/'manifest.json';source.write_text(json.dumps(manifest))
    result=summarize_nuisance_shift(source,tmp_path/'out',repeats=20)
    assert result['runs']==4 and result['long_rows']==32 and result['shift_rows']==24
    shifts=pd.read_csv(tmp_path/'out/ood_environment_shifts.csv')
    spurious=shifts[(shifts.metric=='accuracy') & (shifts.evaluation_environment=='spurious')]
    assert set(spurious.shift_minus_iid.round(8))=={-.4}
    contrasts=json.loads((tmp_path/'out/ood_adjacent_gamma_contrasts.json').read_text())
    accuracy=next(row for row in contrasts if row['evaluation_environment']=='iid' and row['metric']=='accuracy')
    assert accuracy['paired_seeds']==[0,1] and abs(accuracy['mean']-.05)<1e-12
    assert (tmp_path/'out/ood_intervention_shifts.svg').is_file()
