import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
for experiment in sorted((root/"results").iterdir()):
    if not experiment.is_dir():
        continue
    summaries = list(experiment.glob("runs/*/summary.json"))
    trials = list(experiment.glob("trials/*/summary.json"))
    statuses = {}
    for path in summaries:
        value = json.loads(path.read_text())
        statuses[value["status"]] = statuses.get(value["status"], 0)+1
    print(json.dumps(dict(experiment=experiment.name, runs=len(summaries), statuses=statuses,
                         calibration_trials=len(trials),
                         metric_checkpoints=len(list(experiment.glob("runs/*/metrics/*/step_*.json"))),
                         metric_finals=len(list(experiment.glob("runs/*/metrics/*/final.json"))))))
