"""Package complete planned ablations by data condition for remote_metrics workers."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.completion_inventory import DEFAULTS, inventory
from scripts.prepare_primary_inputs import package
from src.training.checkpoints import atomic_json, fingerprint
from research_ext.catalog import ABLATION_PRODUCTION

STUDIES=('width','depth','relevance','swapped','cylinder','small_network','ood')
DATA_KEYS=('dimension','manifold','swap','relevance','relevance_mode','nuisance_condition')


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


def prepare(repo,output,study,index_output=None):
    report=inventory(repo,check_tensors=True)
    conditions,excluded=groups(report,study)
    output=Path(output)
    indexes=[]
    for key,group in sorted(conditions.items()):
        package(repo,output/key,metric_options=ABLATION_PRODUCTION,**group)
        indexes.append(dict(condition=key,index=f'{key}/input_index.json',runs=len(group['rows'])))
    result=dict(schema='feature-topology.ablation-inputs.v1',study=study,
                planned=report['studies'][study]['expected'],conditions=indexes,
                excluded_diverged=excluded,
                scope='Training inputs only; full production metrics remain required')
    atomic_json(output/'study_index.json',result)
    if index_output is not None:
        index_output=Path(index_output)
        portable=[]
        for row in indexes:
            source=output/row['index']
            destination=index_output/f"{row['condition']}.json"
            atomic_json(destination,json.loads(source.read_text()))
            portable.append({**row,'index':destination.name})
        atomic_json(index_output/'study_index.json',{**result,'conditions':portable})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--study',choices=STUDIES,required=True)
    parser.add_argument('--index-output',help='Optional directory for portable committed index copies')
    args=parser.parse_args()
    print(json.dumps(prepare(args.repo,args.output,args.study,args.index_output)))
