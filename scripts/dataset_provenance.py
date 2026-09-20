"""Bind official image source archives to canonical identities and SHA-256."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.checkpoints import atomic_json


EXPECTED_SOURCES = {
    "data/dsprites/dsprites.npz": ("git_blob_sha1", "d98996d3f8ee1550e06b9d4ed78d0beb478910d5"),
    "data/cifar/cifar-10-python.tar.gz": ("md5", "c58f30108f718f92721af3b95e74349a"),
    "data/cifar/cifar-100-python.tar.gz": ("md5", "eb9058c3a382ffc7106e4002c42a8d85"),
}


def file_digest(path, algorithm):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def git_blob_digest(path):
    path = Path(path)
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path, algorithm):
    if algorithm == "git_blob_sha1":
        return git_blob_digest(path)
    if algorithm == "md5":
        return file_digest(path, "md5")
    raise ValueError(f"Unsupported source identity: {algorithm}")


def create_provenance(repo):
    repo = Path(repo).resolve()
    records = []
    for relative, (algorithm, expected) in EXPECTED_SOURCES.items():
        path = repo/relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing official dataset source: {path}")
        actual = source_identity(path, algorithm)
        if actual != expected:
            raise ValueError(f"Official dataset identity mismatch: {relative}")
        records.append({"path": relative, "bytes": path.stat().st_size,
                        "sha256": file_digest(path, "sha256"),
                        "source_identity": {"algorithm": algorithm, "value": actual}})
    return {"schema": "feature-topology.dataset-provenance.v1", "sources": records,
            "scope": "Official source archives only; split/design identities are recorded by each frozen image study."}


def verify_provenance(repo, manifest_path):
    repo, manifest_path = Path(repo).resolve(), Path(manifest_path)
    import json
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "feature-topology.dataset-provenance.v1":
        raise ValueError("Unexpected dataset provenance schema")
    records = {row.get("path"): row for row in manifest.get("sources", [])}
    if set(records) != set(EXPECTED_SOURCES):
        raise ValueError("Dataset provenance source set changed")
    paths = [manifest_path]
    for relative, (algorithm, expected_identity) in EXPECTED_SOURCES.items():
        row, source = records[relative], repo/relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or not source.resolve().is_relative_to(repo) or not source.is_file()
                or row.get("bytes") != source.stat().st_size
                or row.get("sha256") != file_digest(source, "sha256")
                or row.get("source_identity") != {"algorithm": algorithm, "value": expected_identity}
                or source_identity(source, algorithm) != expected_identity):
            raise ValueError(f"Dataset source missing or changed: {relative}")
        paths.append(source)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="results/completion/dataset_provenance.json")
    args = parser.parse_args()
    result = create_provenance(args.repo)
    atomic_json(Path(args.output), result)
    verify_provenance(args.repo, args.output)
    print({"verified_sources": len(result["sources"])})
