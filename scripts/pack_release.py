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
from src.training.checkpoints import atomic_json


class Parts:
    def __init__(self,root,limit,on_part=None,retain_parts=True):
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
                self.stream=self.path.open('xb')
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


def pack(repo,manifest_path,output,part_bytes=1024**3,analysis_manifest=None,
         on_part=None,retain_parts=True):
    repo=Path(repo).resolve()
    manifest=verify_manifest(repo,manifest_path)
    files=dict(manifest['files'])
    if analysis_manifest is not None:
        analysis_path=Path(analysis_manifest).resolve()
        if not analysis_path.is_relative_to(repo):
            raise ValueError('Analysis manifest must be inside the repository')
        analysis=json.loads(analysis_path.read_text())
        if (analysis.get('schema')!='feature-topology.final-analysis.v1'
            or analysis.get('frozen_manifest_sha256')!=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            or not analysis.get('files')):
            raise ValueError('Analysis is not bound to this frozen input manifest')
        for relative,expected in analysis['files'].items():
            source=repo/relative
            if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not source.resolve().is_relative_to(repo) or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest()!=expected
                or relative in files):
                raise ValueError(f'Analysis artifact missing, changed, or conflicting: {relative}')
            files[relative]=expected
        files[analysis_path.relative_to(repo).as_posix()]=hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    sink=Parts(output,part_bytes,on_part=on_part,retain_parts=retain_parts)
    with gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0,compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(files):
                source=repo/relative
                info=tarfile.TarInfo('feature-topology/'+relative)
                info.size=source.stat().st_size
                info.mode=0o644
                with source.open('rb') as stream:
                    archive.addfile(info,stream)
            payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
            info=tarfile.TarInfo('feature-topology/frozen_manifest.json')
            info.size=len(payload)
            info.mode=0o644
            archive.addfile(info,io.BytesIO(payload))
    sink.finish_part()
    result=dict(schema='feature-topology.release-parts.v1',parts=sink.records,
                files=files,
                frozen_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                restore='Concatenate parts in numeric filename order, then extract the resulting tar.gz. Verify frozen_manifest.json with scripts/freeze_results.py utilities.')
    atomic_json(Path(output)/'release_parts.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--analysis-manifest')
    parser.add_argument('--upload-release',
                        help='Upload each verified part to this existing GitHub release')
    args=parser.parse_args()
    uploader=GithubReleaseUploader(args.upload_release) if args.upload_release else None
    result=pack(args.repo,args.manifest,args.output,
                analysis_manifest=args.analysis_manifest,on_part=uploader,
                retain_parts=uploader is None)
    if uploader is not None:
        index=Path(args.output)/'release_parts.json'
        uploader(index,dict(file=index.name,bytes=index.stat().st_size,
                            sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
    print(json.dumps(result,indent=2))
