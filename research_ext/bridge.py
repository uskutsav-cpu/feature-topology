"""Export exact stored weights from upstream MLP checkpoints, with provenance."""
from __future__ import annotations
from pathlib import Path
import re
from .exact import Layer, stored_float
from .io import file_digest


def export_checkpoint(path: str | Path, *, hidden_layers: int | None = None,
                      include_head: bool = False, max_parameters: int = 100000) -> dict:
    import torch
    path = Path(path)
    # Do not silently fall back to general pickle loading.
    state = torch.load(path, map_location="cpu", weights_only=True)
    weights = state.get("model", state)
    if not isinstance(weights, dict):
        raise ValueError("Expected a model state dictionary")
    # Upstream ScaledModel checkpoints contain network.* and initial.* keys.
    prefix = "network." if any(k.startswith("network.hidden.") for k in weights) else ""
    indices = sorted({int(m.group(1)) for key in weights
                      if (m := re.fullmatch(re.escape(prefix) + r"hidden\.(\d+)\.weight", key))})
    if not indices or indices != list(range(len(indices))):
        raise ValueError("Only the upstream consecutive Linear/ReLU MLP architecture is supported")
    count = len(indices) if hidden_layers is None else hidden_layers
    if not 1 <= count <= len(indices):
        raise ValueError("hidden_layers is outside the checkpoint architecture")
    if include_head and count != len(indices):
        raise ValueError("The head can only follow all hidden layers")
    layers, total = [], 0
    names = [(f"{prefix}hidden.{i}", True) for i in indices[:count]]
    if include_head:
        names.append((f"{prefix}head", False))
    for name, relu in names:
        w, b = weights[name+".weight"], weights[name+".bias"]
        if w.ndim != 2 or b.ndim != 1:
            raise ValueError("Expected affine tensor shapes")
        total += w.numel()+b.numel()
        if total > max_parameters:
            raise ValueError("Exact parameter budget exceeded")
        layers.append(Layer(tuple(tuple(stored_float(v) for v in row) for row in w.tolist()),
                            tuple(stored_float(v) for v in b.tolist()), relu))
    return {"layers": [layer.to_dict() for layer in layers],
            "provenance": {"checkpoint_sha256": file_digest(path), "checkpoint_file": path.name,
                           "step": state.get("step"), "hidden_layers": count, "include_head": include_head,
                           "parameter_count": total,
                           "semantics": "Raw network hidden representation, or raw network head when requested. Not centered/scaled model logits.",
                           "arithmetic": "Ideal rational arithmetic on exact stored finite weights; not IEEE floating-point inference."}}
