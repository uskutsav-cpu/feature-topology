"""Collect all sixteen v4 attempts with source, scope, and replay checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flint import arb,ctx
from scripts.recover_release_evidence import digest,load_json,publish
from scripts.remote_rational_fiber_verifier import source_files
from src.metrics.fixed_fiber_verifier import SCOPE,check_circle_cover
from src.metrics.rational_fixed_fiber_verifier import exact_bound,EPSILON_BITS,FACTOR_BITS


def collect(repo,artifacts,output):
    repo,artifacts,output=Path(repo).resolve(),Path(artifacts),Path(output)
    index_path=repo/'configs/completion/primary_inputs_v3.json'
    index=load_json(index_path.read_bytes())
    records={row['run_id']:row for row in index['runs']}
    witnesses=load_json((repo/'results/continuous_fiber_feasibility/manifest.json').read_bytes())['runs']
    eligible={row['run_id']:row for row in witnesses if row['empirical_screen_passed']}
    expected={(rid,theta) for rid in eligible for theta in [1,3,5,7]}
    if len(eligible)!=4 or len(expected)!=16:
        raise ValueError('Different eligible witness grid')
    sources={str(path.relative_to(repo)):digest(path) for path in source_files(repo)}
    protocol_sha=digest(repo/'configs/fixed_fiber_verifier_v4.json')
    files={};summary=[];seen=set()
    ctx.prec=128
    for path in sorted(artifacts.rglob('*.json')):
        result=load_json(path.read_bytes())
        if 'binding' not in result:continue
        binding=result['binding'];rid=binding['run_id'];theta=binding['theta_fraction']
        key=(rid,theta[0])
        if key not in expected or key in seen or theta[1]!=8:
            raise ValueError('Unexpected or duplicate fixed-circle attempt')
        seen.add(key)
        record,witness=records[rid],eligible[rid]
        if (result.get('schema')!='feature-topology.fixed-fiber-verification.v4'
                or binding.get('gamma')!=witness['gamma'] or record['seed']!=0
                or binding.get('checkpoint_sha256')!=witness['checkpoint_sha256']
                or binding['checkpoint_sha256']!=record['files'][witness['checkpoint']]
                or binding.get('input_archive_sha256')!=record['archive_sha256']
                or binding.get('input_index_sha256')!=digest(index_path)
                or binding.get('analysis_data_sha256')!=index['analysis_data']['sha256']
                or binding.get('witness_sha256')!=digest(repo/'results/continuous_fiber_feasibility'/(rid+'.json'))
                or binding.get('source_sha256')!=sources or binding.get('protocol_sha256')!=protocol_sha
                or result.get('theta_fraction')!=theta or result.get('scope')!=SCOPE
                or result.get('arithmetic')!={'python_flint':'0.9.0','precision_bits':128}):
            raise ValueError('Fixed-circle source, arithmetic, or scope binding mismatch')
        receipt_path=path.with_name(path.stem+'.replay.json')
        receipt=load_json(receipt_path.read_bytes())
        positive=result.get('continuous_certificate') is True
        if (receipt.get('schema')!='feature-topology.fixed-fiber-replay.v4'
                or receipt.get('run_id')!=rid or receipt.get('theta_fraction')!=theta
                or receipt.get('result_sha256')!=digest(path)
                or receipt.get('positive_certificate') is not positive
                or receipt.get('standalone_replay_passed') is not positive
                or receipt.get('status')!=('passed' if positive else 'not_a_positive_certificate')):
            raise ValueError('Missing or different standalone replay receipt')
        if positive:
            if result.get('status')!='certified_fixed_nuisance_circle':
                raise ValueError('Positive certificate has a different status')
            check_circle_cover(result['leaf_codes'])
            epsilon=exact_bound(result['epsilon_upper_rational'],EPSILON_BITS)
            factor=exact_bound(result['chord_lower_factor_rational'],FACTOR_BITS)
            if not epsilon<2/arb.pi() or not factor>0:
                raise ValueError('Positive certificate has no positive separation margin')
            coefficients=[arb(float.fromhex(value)) for row in witness['readout_hex'] for value in row]
            norm=sum((value**2 for value in coefficients),arb(0)).sqrt()
            if not factor<=(1-epsilon*arb.pi()/2)/norm:
                raise ValueError('Separation factor differs from the fixed readout witness')
        elif result.get('status') not in {'inconclusive','fixed_witness_rejected'}:
            raise ValueError('Unknown negative/inconclusive attempt status')
        files[path.name]=path.read_bytes();files[receipt_path.name]=receipt_path.read_bytes()
        summary.append(dict(run_id=rid,gamma=witness['gamma'],seed=0,theta_fraction=theta,
            status=result['status'],continuous_certificate=positive,
            result_sha256=digest(path),replay_sha256=digest(receipt_path),
            certified_intervals=len(result.get('leaf_codes',[])),
            epsilon_upper_rational=result.get('epsilon_upper_rational'),
            chord_lower_factor_rational=result.get('chord_lower_factor_rational')))
    if seen!=expected:
        raise ValueError(f'Fixed-circle collection incomplete: {len(seen)}/16')
    manifest=dict(schema='feature-topology.fixed-fiber-collection.v4',expected_attempts=16,
        complete_attempts=16,positive_certificates=sum(row['continuous_certificate'] for row in summary),
        independent_trained_models=4,training_seeds=[0],protocol_sha256=protocol_sha,records=summary,
        scope='Specified sixteen circles in four real-valued saved-weight models; no universal-theta/full-torus or statistical replication claim',
        replay_scope='Every positive certificate replayed in a separate process using the same interval implementation; not a Lean proof')
    files['collection.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    publish(output,files)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--artifacts',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=collect(args.repo,args.artifacts,args.output)
    print(json.dumps({key:manifest[key] for key in ['complete_attempts','positive_certificates','independent_trained_models']}))
