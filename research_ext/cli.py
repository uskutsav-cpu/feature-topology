"""User-facing commands; all write locations are explicit and additive."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .io import atomic_json, read_json
from .catalog import load_catalog, coverage
from .report import analyze
from .exact import (Layer, point, polygon_certificate, certify_box,
                    verify_polygon_certificate, verify_box_certificate)
from .bridge import export_checkpoint
from .formal import export_graph, check_formal
from .workflow import make_plan, execute, GAMMAS


def exact_controls(output: str | Path) -> dict:
    output=Path(output)
    square=[point([-1,-1]),point([1,-1]),point([1,1]),point([-1,1])]
    def layer(w,b,relu=False): return Layer(tuple(point(row) for row in w),point(b),relu)
    designs={'identity':([],1,True),
             'scaled_rotated':([layer([[0,-3],[3,0]],[2,7])],1,True),
             'projection':([layer([[1,0]],[0])],0,False),
             'constant':([layer([[0,0]],[0])],0,False),
             'relu_same_cycle_rank_collision':([layer([[1,0],[0,1]],[0,0],True)],1,False),
             'signed_relu_embedding':([layer([[1,0],[-1,0],[0,1],[0,-1]],[0,0,0,0],True)],1,True)}
    results={}
    for name,(layers,beta,injective) in designs.items():
        cert=polygon_certificate(square,layers)
        if cert['graph']['beta1']!=beta or cert['injective_on_polygon_subset']!=injective or not verify_polygon_certificate(cert):
            raise ArithmeticError(f'Exact control failed: {name}')
        atomic_json(output/f'{name}.json',cert)
        results[name]={'beta0':cert['graph']['beta0'],'beta1':cert['graph']['beta1'],
                       'injective_on_polygon_subset':cert['injective_on_polygon_subset'],
                       'collision':cert['collision'],'python_exact_replay_passed':True,'lean_verified':False}
    box=certify_box([layer([[2,0],[0,3]],[0,0],True)],point([1,1]),point([2,2]))
    if not verify_box_certificate(box): raise ArithmeticError('Box control failed')
    atomic_json(output/'positive_box.json',box)
    results['positive_box']={'status':box['status'],'python_exact_replay_passed':True,'lean_verified':False}
    summary={'status':'exact_controls_passed','controls':results,
             'scope':'Analytic controls, not observations from the production training sweep.'}
    atomic_json(output/'control_summary.json',summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Auditable analysis and exact small-network tooling')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['audit','analyze']:
        c=sub.add_parser(name)
        c.add_argument('--roots',nargs='+',required=True)
        c.add_argument('--output',required=True)
        c.add_argument('--profile')
        c.add_argument('--config',default='configs/research_ext/analysis.json')
        if name=='analyze':
            c.add_argument('--plots',action='store_true')
    c=sub.add_parser('controls');c.add_argument('--output',required=True)
    c=sub.add_parser('export-checkpoint');c.add_argument('--checkpoint',required=True)
    c.add_argument('--output',required=True);c.add_argument('--hidden-layers',type=int)
    c.add_argument('--include-head',action='store_true')
    for name in ['certify-polygon','certify-box']:
        c=sub.add_parser(name)
        g=c.add_mutually_exclusive_group(required=True)
        g.add_argument('--network');g.add_argument('--checkpoint')
        c.add_argument('--hidden-layers',type=int)
        c.add_argument('--domain',required=True)
        c.add_argument('--output',required=True)
        if name=='certify-polygon':c.add_argument('--max-segments',type=int,default=512)
    c=sub.add_parser('verify-certificate');c.add_argument('certificate')
    c=sub.add_parser('export-lean');c.add_argument('certificate')
    c.add_argument('--output',default='formal/FeatureTopology/Generated.lean')
    c=sub.add_parser('check-formal');c.add_argument('--root',default='formal')
    c.add_argument('--output',required=True);c.add_argument('--timeout',type=int,default=180)
    c=sub.add_parser('plan');c.add_argument('--repo',default='.')
    c.add_argument('--suite',choices=['primary','ablations','formal','images','cifar'],default='primary')
    c.add_argument('--output',required=True);c.add_argument('--image-device',choices=['cpu','mps'],default='cpu')
    c=sub.add_parser('execute-plan');c.add_argument('plan')
    c.add_argument('--only',nargs='+');c.add_argument('--state-root')
    c.add_argument('--task-timeout',type=float)
    c=sub.add_parser('nested-metrics');c.add_argument('--root',required=True)
    c=sub.add_parser('nested-analyze');c.add_argument('--root',required=True)
    c.add_argument('--profile',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('noise-checkpoint');c.add_argument('--checkpoint',required=True);c.add_argument('--output',required=True)
    c.add_argument('--layer',type=int);c.add_argument('--train-points',type=int,default=2000);c.add_argument('--test-points',type=int,default=1000)
    c.add_argument('--repeats',type=int,default=5)
    c=sub.add_parser('audit-images');c.add_argument('--root',required=True);c.add_argument('--output',required=True)
    c=sub.add_parser('smoke');c.add_argument('--output',required=True)
    return p


def main(argv=None) -> int:
    a=parser().parse_args(argv)
    try:
        if a.command in {'audit','analyze'}:
            config=read_json(a.config)
            if a.command=='audit':
                c=load_catalog(a.roots,profile=a.profile)
                result=coverage(c,config['gammas'],config['seeds'])
                atomic_json(a.output,result)
            else:
                result=analyze(a.roots,a.output,target=config['target_loss'],profile=a.profile,
                    gammas=config['gammas'],seeds=config['seeds'],repeats=config['bootstrap_repeats'],
                    transition_repeats=config['transition_repeats'],rules=config['rules'],make_plots=a.plots)
        elif a.command=='controls':result=exact_controls(a.output)
        elif a.command=='export-checkpoint':
            result=export_checkpoint(a.checkpoint,hidden_layers=a.hidden_layers,include_head=a.include_head)
            atomic_json(a.output,result)
        elif a.command in {'certify-polygon','certify-box'}:
            network=read_json(a.network) if a.network else export_checkpoint(a.checkpoint,hidden_layers=a.hidden_layers)
            layers=[Layer.from_dict(v) for v in network['layers']]
            domain=read_json(a.domain)
            if a.command=='certify-polygon':
                result=polygon_certificate([point(v) for v in domain['vertices']],layers,max_segments=a.max_segments)
            else:
                result=certify_box(layers,point(domain['lower']),point(domain['upper']))
            if 'provenance' in network: result['provenance']=network['provenance']
            atomic_json(a.output,result)
        elif a.command=='verify-certificate':
            cert=read_json(a.certificate)
            valid=(verify_polygon_certificate(cert) if cert.get('schema')=='feature-topology.exact-polygon.v1'
                   else verify_box_certificate(cert))
            result={'python_exact_replay_passed':valid,'lean_verified':False}
            print(json.dumps(result,indent=2))
            return 0 if valid else 2
        elif a.command=='export-lean':result=export_graph(read_json(a.certificate),a.output)
        elif a.command=='check-formal':
            result=check_formal(a.root,a.output,timeout=a.timeout)
            print(json.dumps(result,indent=2))
            return 0 if result['lean_verified'] else 2
        elif a.command=='plan':
            result=make_plan(a.repo,suite=a.suite,image_device=a.image_device);atomic_json(a.output,result)
        elif a.command=='execute-plan':
            result=execute(read_json(a.plan),state_root=a.state_root,only=a.only,timeout=a.task_timeout)
        elif a.command in {'nested-metrics','nested-analyze'}:
            root=Path(a.root).resolve()
            groups=read_json(root/'manifest.json')
            result=[]
            for group in groups:
                directory=Path(group['path']).resolve()
                if not directory.is_relative_to(root):
                    raise ValueError('Ablation manifest points outside the selected study')
                if a.command=='nested-metrics':
                    subprocess.run([sys.executable,'scripts/compute_metrics.py','--runs',str(directory/'runs'),
                                    '--all-checkpoints'],check=True)
                    result.append({'group':directory.name,'status':'metric_command_completed'})
                else:
                    gammas=sorted({r['gamma'] for r in group['runs']})
                    seeds=sorted({r['seed'] for r in group['runs']})
                    result.append(analyze([directory],Path(a.output)/directory.name,profile=a.profile,
                                          gammas=gammas,seeds=seeds))
        elif a.command=='noise-checkpoint':
            from .noise import checkpoint_noise
            result=checkpoint_noise(a.checkpoint,a.output,layer=a.layer,train_points=a.train_points,
                                    test_points=a.test_points,repeats=a.repeats)
        elif a.command=='audit-images':
            from .image_audit import audit_images
            result=audit_images(a.root)
            atomic_json(a.output,result)
        elif a.command=='smoke':
            from .smoke import run_smoke
            result=run_smoke(a.output)
        else:raise ValueError('Unknown command')
        # Large certificates stay in the output file rather than flooding the terminal.
        if a.command in {'certify-polygon','certify-box','export-checkpoint'}:
            print(json.dumps({'output':a.output,'status':result.get('status','written'),
                              'evidence_level':result.get('evidence_level','export_only'),'lean_verified':False},indent=2))
        else:print(json.dumps(result,indent=2))
        return 0
    except (OSError,ValueError,KeyError,RuntimeError,ArithmeticError,subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
