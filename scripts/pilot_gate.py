"""Operational go/no-go gate; not a confirmatory hypothesis test."""
import argparse
import json
from pathlib import Path
import numpy as np


def gate(root):
    root = Path(root)
    rows = []
    for p in root.glob("runs/*/metrics/*/final.json"):
        d = json.loads(p.read_text())
        summary = json.loads((p.parents[2]/"summary.json").read_text())
        last = d["layers"][-1]
        rows.append(dict(gamma=d["gamma"], seed=d["seed"], profile=p.parent.name,
                         converged=summary["status"] == "converged",
                         training_loss=summary["history"][-1]["training_loss"],
                         ntk_drift=d["ntk_drift"], cka_drift=last["cka_drift"],
                         local_q01=last["jacobian"]["normalized_q01"],
                         nuisance_cosine=last["probes"]["mlp"].get("angular_cosine")))
    profiles = {r["profile"] for r in rows}
    if len(profiles) != 1:
        raise RuntimeError("Gate requires exactly one pilot metric profile")
    expected = {(g, s) for g in [.03125, .5, 1., 16., 128.] for s in range(3)}
    observed = {(r["gamma"], r["seed"]) for r in rows}
    if observed != expected:
        raise RuntimeError(f"Pilot metrics incomplete: {len(observed)}/15")
    low = [r for r in rows if r["gamma"] == .03125]
    high = [r for r in rows if r["gamma"] == 128.]
    diagnostics = dict(all_converged=all(r["converged"] for r in rows),
                       low_gamma_ntk_median=float(np.median([r["ntk_drift"] for r in low])),
                       high_gamma_ntk_median=float(np.median([r["ntk_drift"] for r in high])),
                       high_gamma_cka_median=float(np.median([r["cka_drift"] for r in high])),
                       high_gamma_nuisance_cosine=float(np.median([r["nuisance_cosine"] for r in high])))
    proceed = (diagnostics["all_converged"] and diagnostics["low_gamma_ntk_median"] < .1 and
               diagnostics["high_gamma_ntk_median"] > 1 and diagnostics["high_gamma_cka_median"] > .05)
    value = dict(proceed_to_main=proceed, diagnostics=diagnostics, runs=rows,
                 interpretation="Operational pilot gate establishes a measurable regime contrast and optimization comparability. It neither requires nor certifies information destruction.",
                 thresholds="Exploratory engineering checks: low NTK <0.1, high NTK >1, high CKA drift >0.05; these were not preregistered statistical tests.")
    (root/"gate.json").write_text(json.dumps(value, indent=2)+"\n")
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", default="results/pilot")
    a = p.parse_args(); print(json.dumps(gate(a.root), indent=2))
