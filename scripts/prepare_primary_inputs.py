"""Package verified primary checkpoints and fixed analysis arrays for CI workers."""
import argparse
import gzip
import json
from pathlib import Path
import platform
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.completion_inventory import inspect_run,sha256
from src.training.train import data_for
from src.data.torus import dataset
from src.training.checkpoints import atomic_json


def package(repo,output):
    repo,output=Path(repo),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((repo/'results/main/manifest.json').read_text())
    expected={(2.**i,s) for i in range(-5,8) for s in range(10)}
    if len(rows)!=130 or {(r['gamma'],r['seed']) for r in rows}!=expected:
        raise ValueError('Primary training manifest is incomplete')
    data_config=dict(dimension=16,manifold='torus',swap=False,relevance=0.,relevance_mode='periodic')
    data_path=output/'analysis_data.npz'
    if not data_path.exists():
        _,_,(gx,gy,z,q)=data_for(data_config)
        px,py,pz,_=dataset(5000,seed=4001,**data_config)
        tx,ty,tz,_=dataset(2000,seed=4002,**data_config)
        np.savez_compressed(data_path,data_config=json.dumps(data_config,sort_keys=True),
                            **{k:v.numpy() for k,v in locals().copy().items()
                               if k in ['gx','gy','z','q','px','py','pz','tx','ty','tz']})
    records=[]
    for row in sorted(rows,key=lambda r:(r['gamma'],r['seed'])):
        run=repo/'results/main/runs'/row['run_id']
        verified=inspect_run(run,check_tensors=True)
        actual={k:verified['config'].get(k,v) for k,v in data_config.items()}
        if actual!=data_config:
            raise ValueError('Nonprimary condition in the input manifest')
        files=sorted([run/'config.json',run/'summary.json',run/'final.pt',*run.glob('step_*.pt')])
        hashes={p.name:sha256(p) for p in files}
        archive=output/(run.name+'.tar.gz')
        if not archive.exists():
            temporary=archive.with_suffix('.partial')
            with temporary.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=1) as zipped:
                with tarfile.open(fileobj=zipped,mode='w|') as tar:
                    for path in files:
                        info=tarfile.TarInfo(run.name+'/'+path.name)
                        info.size=path.stat().st_size
                        info.mode=0o644
                        with path.open('rb') as source:tar.addfile(info,source)
            temporary.replace(archive)
        records.append(dict(**row,asset=archive.name,archive_sha256=sha256(archive),
                            files=hashes,config=verified['config']))
        print(json.dumps(dict(run_id=run.name,bytes=archive.stat().st_size)),flush=True)
    result=dict(schema='feature-topology.primary-inputs.v1',runs=records,
                analysis_data=dict(asset=data_path.name,sha256=sha256(data_path),config=data_config),
                export_environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                scope='Frozen training inputs for production analysis; not a final scientific-results release')
    atomic_json(output/'input_index.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    package(args.repo,args.output)
