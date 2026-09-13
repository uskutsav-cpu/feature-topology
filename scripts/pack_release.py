"""Deterministic, size-bounded release parts from verified frozen inputs only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.freeze_results import verify_manifest
from src.training.checkpoints import atomic_json


class Parts:
    def __init__(self,root,limit):
        self.root,self.limit=Path(root),limit
        self.root.mkdir(parents=True,exist_ok=True)
        self.stream=None
        self.records=[]
        self.size=0

    def finish_part(self):
        if self.stream is not None:
            self.stream.close()
            self.records.append(dict(file=self.path.name,bytes=self.size,sha256=self.hash.hexdigest()))
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


def pack(repo,manifest_path,output,part_bytes=1024**3):
    repo=Path(repo).resolve()
    manifest=verify_manifest(repo,manifest_path)
    sink=Parts(output,part_bytes)
    with gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0,compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(manifest['files']):
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
                frozen_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                restore='Concatenate parts in numeric filename order, then extract the resulting tar.gz. Verify frozen_manifest.json with scripts/freeze_results.py utilities.')
    atomic_json(Path(output)/'release_parts.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(pack(args.repo,args.manifest,args.output),indent=2))
