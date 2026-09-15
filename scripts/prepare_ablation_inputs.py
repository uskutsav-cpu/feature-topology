"""Package complete planned ablations by data condition for remote_metrics workers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import DEFAULTS, inventory
from scripts.prepare_primary_inputs import package
from src.training.checkpoints import atomic_json, fingerprint

STUDIES=('width','depth','relevance','swapped','cylinder','small_network')
DATA_KEYS=('dimension','manifold','swap','relevance','relevance_mode')


def groups(report,study):
    plan=report['studies'][study]
    if plan['accounted']!=plan['expected'] or plan['missing']:
        raise ValueError(f'{study} training is incomplete')
    records={r['run_id']:r for r in report['validated_runs']}
    selected={row['run_id'] for row in plan['runs']}
    result={}
    excluded=[]
    for run_id in sorted(selected):
        record=records[run_id]
        if record['status']=='diverged':
            excluded.append(record)
            continue
        config={**DEFAULTS,**record['config']}
        condition={k:config[k] for k in DATA_KEYS}
        key=fingerprint(condition)
        group=result.setdefault(key,dict(data_config=condition,rows=[],run_paths={}))
        group['rows'].append(dict(run_id=run_id,gamma=config['gamma'],seed=config['seed']))
        group['run_paths'][run_id]=record['path']
    return result,excluded


def prepare(repo,output,study):
    report=inventory(repo,check_tensors=True)
    conditions,excluded=groups(report,study)
    output=Path(output)
    indexes=[]
    for key,group in sorted(conditions.items()):
        package(repo,output/key,**group)
        indexes.append(dict(condition=key,index=f'{key}/input_index.json',runs=len(group['rows'])))
    result=dict(schema='feature-topology.ablation-inputs.v1',study=study,
                planned=report['studies'][study]['expected'],conditions=indexes,
                excluded_diverged=excluded,
                scope='Training inputs only; full production metrics remain required')
    atomic_json(output/'study_index.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--study',choices=STUDIES,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.repo,args.output,args.study)))
