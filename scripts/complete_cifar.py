"""Queue the original CIFAR10/CIFAR100 scope after the dSprites GPU study."""
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.checkpoints import atomic_json
from scripts.run_cifar import load_data


def main():
    repo=Path(__file__).resolve().parents[1]
    output=repo/"results/completion"
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ["CIFAR10","CIFAR100"]:
        data=load_data(name,str(repo/"data/cifar"),download=True)
        sizes={split:len(pair[0]) for split,pair in data.items()}
        if sizes!={"train":45000,"validation":5000,"test":10000}:
            raise ValueError(f"Unexpected {name} split sizes")
        rows.append(dict(dataset=name,stage="data_validated",sizes=sizes))
        del data
        atomic_json(output/"cifar_queue.json",rows)
    while True:
        manifest=repo/"results/dsprites/manifest.json"
        if manifest.exists():
            values=json.loads(manifest.read_text())
            if len(values)==35 and len({(v["gamma"],v["seed"]) for v in values})==35:
                break
        time.sleep(30)
    for name in ["CIFAR10","CIFAR100"]:
        command=[sys.executable,"scripts/run_cifar.py","--dataset",name,
                 "--output",f"results/{name.lower()}","--device","mps"]
        row=dict(dataset=name,stage="running",command=command,started=time.time())
        rows.append(row)
        atomic_json(output/"cifar_queue.json",rows)
        with (output/f"{name.lower()}.log").open("a") as log:
            result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,stage="command_completed" if result.returncode==0 else "failed",
                   finished=time.time())
        atomic_json(output/"cifar_queue.json",rows)
    return int(any(r.get("returncode",0) for r in rows))


if __name__=="__main__":
    sys.exit(main())
