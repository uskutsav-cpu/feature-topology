import itertools
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def main():
    root = Path(__file__).resolve().parents[1]/"configs"
    pilot = json.loads((root/"synthetic/calibration_pilot.json").read_text())
    gammas = [2.**i for i in range(-5, 8)]
    representative = [.125, .5, 1., 4., 16., 64., 128.]
    write(root/"synthetic/calibration_main.json", {**pilot, "gammas":gammas})
    plans = {
        "main": [dict(gamma=g, seed=s, width=256, depth=4, manifold="torus") for g,s in itertools.product(gammas, range(10))],
        "width": [dict(width=w, gamma=g, seed=s, depth=4, manifold="torus") for w,g,s in itertools.product([64,128,256,512,1024], representative, range(5))],
        "depth": [dict(depth=d, gamma=g, seed=s, width=256, manifold="torus") for d,g,s in itertools.product([2,4,6,8], representative, range(5))],
        "cylinder": [dict(gamma=g, seed=s, width=256, depth=4, manifold="cylinder") for g,s in itertools.product(representative, range(5))],
        "swapped": [dict(gamma=g, seed=s, width=256, depth=4, manifold="torus", swap=True) for g,s in itertools.product(representative, range(5))],
        "relevance": [dict(gamma=g, seed=s, width=256, depth=4, manifold="torus", relevance=r, relevance_mode="periodic") for g,s,r in itertools.product(representative, range(5), [0,.05,.1,.25,.5,1])],
        "small_network": [dict(gamma=g, seed=s, width=w, depth=d, manifold="torus") for g,s,w,d in itertools.product([.125,1.,16.,128.], range(5), [8,16,32], [2,4])],
    }
    for name, runs in plans.items():
        write(root/"sweeps"/(name+".json"), dict(name=name, runs=runs,
              requires="Passed pilot; separately frozen per-architecture/per-task calibration; deduplicate shared main runs"))
    write(root/"cifar/resnet18.json", dict(datasets=["CIFAR10","CIFAR100"], gammas=representative,
          seeds=list(range(5)), architecture="ResNet18", batch_size=128, epochs=100,
          requires="Completed controlled experiments and independent LR calibration", claims="Geometry only; no known latent topology"))
    write(root/"dsprites/cnn.json", dict(task="shape", nuisance="orientation", architecture="CNN",
          gammas=representative, seeds=list(range(5)),
          requires="Official dSprites archive; stratified factor design; acknowledge shape rotational symmetries"))


if __name__ == "__main__":
    main()
