"""Deterministic, size-bounded release parts from verified frozen inputs only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.freeze_results import verify_manifest
from scripts.recover_release_evidence import load_json
from src.training.checkpoints import atomic_json


class Parts:
    def __init__(self,root,limit,on_part=None,retain_parts=True):
        if type(limit) is not int or limit<=0:
            raise ValueError('Release part size must be a positive integer')
        self.root,self.limit=Path(root),limit
        self.root.mkdir(parents=True,exist_ok=True)
        self.on_part,self.retain_parts=on_part,retain_parts
        self.stream=None
        self.records=[]
        self.size=0

    def finish_part(self):
        if self.stream is not None:
            self.stream.close()
            record=dict(file=self.path.name,bytes=self.size,sha256=self.hash.hexdigest())
            if self.on_part is not None:
                self.on_part(self.path,record)
            self.records.append(record)
            if not self.retain_parts:
                self.path.unlink()
            self.stream=None

    def write(self,data):
        size=len(data)
        while data:
            if self.stream is None:
                self.path=self.root/f'feature-topology.tar.gz.part{len(self.records):03d}'
                # Local packages fail closed on an occupied output path.  A
                # streamed retry may replace only its own generated part left
                # behind by a failed upload; source research artifacts never
                # use this filename or pass through this branch.
                self.stream=self.path.open('xb' if self.retain_parts else 'wb')
                self.hash=hashlib.sha256()
                self.size=0
            block=data[:self.limit-self.size]
            self.stream.write(block)
            self.hash.update(block)
            self.size+=len(block)
            data=data[len(block):]
            if self.size==self.limit:
                self.finish_part()
        return size

    def flush(self):
        if self.stream is not None:
            self.stream.flush()


class GithubReleaseUploader:
    """Idempotently publish a part and verify GitHub's server-side digest."""
    def __init__(self,tag):
        if not re.fullmatch(r"[0-9A-Za-z._-]+",tag):
            raise ValueError("Unsafe GitHub release tag")
        self.tag=tag
        self.assets=self._assets()

    def _assets(self):
        result=subprocess.run(
            ["gh","release","view",self.tag,"--json","assets"],
            check=True,text=True,capture_output=True)
        return {row["name"]:row for row in json.loads(result.stdout)["assets"]}

    @staticmethod
    def _matches(row,record):
        return (row.get("state")=="uploaded"
                and row.get("size")==record["bytes"]
                and row.get("digest")==f"sha256:{record['sha256']}")

    def __call__(self,path,record):
        path=Path(path)
        existing=self.assets.get(record["file"])
        if existing is not None:
            if not self._matches(existing,record):
                raise ValueError(f"Conflicting release asset: {record['file']}")
            return
        subprocess.run(["gh","release","upload",self.tag,str(path)],check=True)
        self.assets=self._assets()
        uploaded=self.assets.get(record["file"])
        if uploaded is None or not self._matches(uploaded,record):
            raise ValueError(f"Uploaded release asset failed digest verification: {record['file']}")


class HashingReader:
    """Check the bytes actually added to the archive, not only an earlier stat."""
    def __init__(self,stream):
        self.stream=stream
        self.hash=hashlib.sha256()
        self.bytes=0

    def read(self,size=-1):
        data=self.stream.read(size)
        self.hash.update(data)
        self.bytes+=len(data)
        return data


def add_derived_file(repo,files,relative,expected):
    source=repo/relative
    if (relative=='frozen_manifest.json' or Path(relative).is_absolute() or '..' in Path(relative).parts
            or not source.resolve().is_relative_to(repo) or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest()!=expected
            or relative in files):
        raise ValueError(f'Analysis artifact missing, changed, or conflicting: {relative}')
    files[relative]=expected


def additional_analysis_files(repo,files,path,frozen,freeze_sha):
    path=Path(path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError('Additional analysis manifest must be inside the repository')
    payload=path.read_bytes()
    analysis=load_json(payload)
    if (analysis.get('schema')!='feature-topology.additional-research-summary.v1'
            or analysis.get('combined_freeze_sha256')!=freeze_sha
            or frozen.get('additional_research',{}).get('schema')!='feature-topology.additional-research-freeze.v1'
            or analysis.get('replication',{}).get('complete_models')!=36
            or analysis.get('replication',{}).get('complete_domains')!=144
            or analysis.get('production_noise',{}).get('unique_models')!=305
            or analysis.get('production_noise_freeze_audit',{}).get('source_bound_same_host_replays_checked') is not True
            or not analysis.get('input_sha256')):
        raise ValueError('Additional analysis is not bound to this complete combined freeze')
    original_path=frozen.get('original_freeze_path')
    original_sha=frozen.get('original_freeze_sha256')
    if (not isinstance(original_path,str) or Path(original_path).is_absolute()
            or '..' in Path(original_path).parts or not original_sha
            or frozen['files'].get(original_path)!=original_sha):
        raise ValueError('Combined release has no bound original-study freeze lineage')
    original=verify_manifest(repo,repo/original_path)
    if (original.get('additional_research') is not None
            or any(frozen['files'].get(name)!=expected for name,expected in original['files'].items())):
        raise ValueError('Original-study frozen inputs are missing or changed in the combined release')
    for relative,expected in analysis['input_sha256'].items():
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or frozen['files'].get(relative)!=expected):
            raise ValueError('Additional analysis consumed an input outside the frozen manifest')
    figures,tables=analysis.get('figure_sha256',{}),analysis.get('table_sha256',{})
    if (not figures or not {'replication_models.csv','replication_domains.csv','noise_metrics_long.csv',
                            'noise_seed_confidence_intervals.csv','noise_relevance_paired_contrasts.csv'}<=set(tables)
            or set(figures)&set(tables)):
        raise ValueError('Additional analysis figure/table index is incomplete or conflicting')
    for name,expected in {**figures,**tables}.items():
        if (not isinstance(name,str) or Path(name).name!=name or name in {'','.','..'}
                or (name in figures and Path(name).suffix not in {'.png','.svg'})
                or (name in tables and Path(name).suffix!='.csv')):
            raise ValueError('Additional analysis artifact name escapes its output directory')
        add_derived_file(repo,files,(path.parent/name).relative_to(repo).as_posix(),expected)
    relative=path.relative_to(repo).as_posix()
    if relative=='frozen_manifest.json' or relative in files:
        raise ValueError('Additional analysis manifest conflicts with another frozen artifact')
    files[relative]=hashlib.sha256(payload).hexdigest()


def pack(repo,manifest_path,output,part_bytes=1024**3,analysis_manifest=None,
         on_part=None,retain_parts=True,additional_analysis_manifest=None):
    repo=Path(repo).resolve()
    manifest=verify_manifest(repo,manifest_path)
    # Analysis receipts refer to these exact bytes; JSON reformatting changes
    # that identity even when its decoded dictionary remains equal.
    manifest_payload=Path(manifest_path).read_bytes()
    if load_json(manifest_payload)!=manifest:
        raise ValueError('Frozen manifest changed during release preparation')
    freeze_sha=hashlib.sha256(manifest_payload).hexdigest()
    files=dict(manifest['files'])
    if 'frozen_manifest.json' in files:
        raise ValueError('Frozen input conflicts with the reserved release manifest path')
    if analysis_manifest is not None:
        analysis_path=Path(analysis_manifest).resolve()
        if not analysis_path.is_relative_to(repo):
            raise ValueError('Analysis manifest must be inside the repository')
        analysis_payload=analysis_path.read_bytes()
        analysis=load_json(analysis_payload)
        if (analysis.get('schema')!='feature-topology.final-analysis.v1'
            or analysis.get('frozen_manifest_sha256')!=freeze_sha
            or not analysis.get('files')):
            raise ValueError('Analysis is not bound to this frozen input manifest')
        for relative,expected in analysis['files'].items():
            add_derived_file(repo,files,relative,expected)
        relative=analysis_path.relative_to(repo).as_posix()
        if relative=='frozen_manifest.json' or relative in files:
            raise ValueError('Analysis manifest conflicts with another frozen artifact')
        files[relative]=hashlib.sha256(analysis_payload).hexdigest()
    if additional_analysis_manifest is not None:
        additional_analysis_files(repo,files,additional_analysis_manifest,manifest,freeze_sha)
    sink=Parts(output,part_bytes,on_part=on_part,retain_parts=retain_parts)
    with gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0,compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(files):
                source=repo/relative
                if not source.resolve().is_relative_to(repo):
                    raise ValueError('Packaged input changed to a path outside the repository')
                info=tarfile.TarInfo('feature-topology/'+relative)
                info.size=source.stat().st_size
                info.mode=0o644
                with source.open('rb') as stream:
                    reader=HashingReader(stream)
                    archive.addfile(info,reader)
                    if reader.bytes!=info.size or reader.hash.hexdigest()!=files[relative]:
                        raise ValueError('Packaged input changed during read: '+relative)
            payload=manifest_payload
            info=tarfile.TarInfo('feature-topology/frozen_manifest.json')
            info.size=len(payload)
            info.mode=0o644
            archive.addfile(info,io.BytesIO(payload))
    sink.finish_part()
    result=dict(schema='feature-topology.release-parts.v1',parts=sink.records,
                files=files,
                frozen_manifest_sha256=freeze_sha,
                additional_analysis_included=additional_analysis_manifest is not None,
                restore='Concatenate parts in numeric filename order, then extract the resulting tar.gz. Verify frozen_manifest.json with scripts/freeze_results.py utilities.')
    atomic_json(Path(output)/'release_parts.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--analysis-manifest')
    parser.add_argument('--additional-analysis-manifest')
    parser.add_argument('--upload-release',
                        help='Upload each verified part to this existing GitHub release')
    args=parser.parse_args()
    uploader=GithubReleaseUploader(args.upload_release) if args.upload_release else None
    result=pack(args.repo,args.manifest,args.output,
                analysis_manifest=args.analysis_manifest,on_part=uploader,
                retain_parts=uploader is None,additional_analysis_manifest=args.additional_analysis_manifest)
    if uploader is not None:
        index=Path(args.output)/'release_parts.json'
        uploader(index,dict(file=index.name,bytes=index.stat().st_size,
                            sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
    print(json.dumps(result,indent=2))
