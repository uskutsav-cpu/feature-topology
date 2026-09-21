import json
import pandas as pd

from research_ext.report import METRICS
from scripts.relevance_statistics import summarize_relevance


def test_relevance_statistics_preserve_seed_pairing(tmp_path):
    groups=[]
    for relevance in [0.,.5,1.]:
        directory=tmp_path/f'lambda-{relevance}'
        directory.mkdir()
        rows=[]
        for gamma in [.5,1.]:
            for seed in [0,1]:
                row=dict(run_id=f'{relevance}-{gamma}-{seed}',gamma=gamma,seed=seed,layer=1)
                row.update({metric:gamma+seed/10+2*relevance for metric in METRICS})
                rows.append(row)
        pd.DataFrame(rows).to_csv(directory/'matched_risk.csv',index=False)
        groups.append((relevance,directory))
    result=summarize_relevance(groups,tmp_path/'out',repeats=20)
    assert result['relevance_levels']==[0.,.5,1.] and result['runs']==12
    contrasts=json.loads((tmp_path/'out/relevance_paired_contrasts.json').read_text())
    row=next(x for x in contrasts if x['comparison']=='baseline' and x['relevance_right']==1.
             and x['gamma']==.5 and x['metric']=='test_accuracy')
    assert row['paired_seeds']==[0,1] and abs(row['mean']-2.)<1e-12
    assert (tmp_path/'out/relevance_regime_map.svg').is_file()
