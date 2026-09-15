"""Replay exact stored-weight certificates on an explicit rational polygon."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_ext.bridge import export_checkpoint
from research_ext.exact import Layer, point, polygon_certificate, verify_polygon_certificate
from research_ext.io import atomic_json, digest, file_digest

# Ten exact unit-circle points. Edges are straight chords, not circular arcs.
VERTICES = [["1","0"],["3/5","4/5"],["0","1"],["-3/5","4/5"],
            ["-4/5","3/5"],["-1","0"],["-3/5","-4/5"],["0","-1"],
            ["3/5","-4/5"],["4/5","-3/5"]]


def run(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(root.glob("runs/*/final.pt")):
        summary = json.loads((path.parent / "summary.json").read_text())
        for depth in range(1, summary["config"]["depth"] + 1):
            target = output / f"{path.parent.name}_layer{depth}.json"
            checkpoint_hash = file_digest(path)
            if target.exists():
                cert = json.loads(target.read_text())
                if cert["provenance"]["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError("Cached certificate belongs to different checkpoint bytes")
            else:
                network = export_checkpoint(path, hidden_layers=depth)
                cert = polygon_certificate([point(v) for v in VERTICES],
                                           [Layer.from_dict(v) for v in network["layers"]])
                cert["provenance"] = network["provenance"]
                cert["training_config"] = summary["config"]
                cert["domain_note"] = "Explicit rational inscribed polygon; no smooth-circle or torus certificate."
                atomic_json(target, cert)
            if not verify_polygon_certificate(cert):
                raise ValueError(f"Exact replay failed: {target}")
            records.append(dict(run_id=path.parent.name, layer=depth,
                                gamma=summary["config"]["gamma"], seed=summary["config"]["seed"],
                                checkpoint_sha256=checkpoint_hash,
                                certificate_file=target.name, certificate_sha256=file_digest(target),
                                graph=cert["graph"], exact_replay_passed=True))
            atomic_json(output / "manifest.json", dict(records=records, domain=VERTICES,
                        complete=False, scope="Ideal rational arithmetic on exact stored trained weights"))
            print(json.dumps({k:v for k,v in records[-1].items() if k != "graph"}), flush=True)
    atomic_json(output / "manifest.json", dict(records=records, domain=VERTICES, complete=True,
                checkpoint_count=len(list(root.glob("runs/*/final.pt"))),
                scope="Ideal rational arithmetic on exact stored trained weights; Python exact replay, not a Lean proof of the network-to-graph construction."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.output)
