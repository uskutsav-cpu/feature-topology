import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models.cnn import CNN
from src.training.checkpoints import atomic_json


torch.set_num_threads(2)
results = {}
for device in ["cpu"]+(["mps"] if torch.backends.mps.is_available() else []):
    torch.manual_seed(5678)
    model = CNN().to(device)
    x = torch.randn(64, 1, 24, 24, device=device)
    y = torch.arange(64, device=device) % 10
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for i in range(55):
        if i == 5:
            if device == "mps": torch.mps.synchronize()
            started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); optimizer.step()
    if device == "mps": torch.mps.synchronize()
    results[device] = (time.monotonic()-started)/50
atomic_json(Path(__file__).resolve().parents[1]/"results/device_benchmark.json", results)
print(json.dumps(results))
