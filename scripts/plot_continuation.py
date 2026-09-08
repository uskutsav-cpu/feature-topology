"""Plot measured clean/noisy ridge probe scores; error bars are seed SD."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/continuation')
    args = parser.parse_args()
    root = Path(args.root)
    noise = pd.read_csv(root/'noise_metrics.csv')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for level, label in [
        (0., 'Clean representation'),
        (.01, 'Noise norm: 1% of training representation RMS'),
        (.1, 'Noise norm: 10% of training representation RMS'),
    ]:
        scores = noise[noise.noise_level == level].groupby('gamma').angular_cosine
        mean, sd = scores.mean(), scores.std()
        ax.errorbar(mean.index, mean, yerr=sd, marker='o', capsize=3, label=label)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Feature-learning strength gamma')
    ax.set_ylabel('Held-out nuisance angular cosine')
    ax.set_title('Clean information retention versus noise sensitivity')
    ax.set_ylim(-.05, 1.05)
    ax.grid(True, alpha=.25)
    ax.legend(loc='lower left', fontsize=9)
    fig.text(.5, .015,
        '39 networks; 13 gamma values x 3 new seeds; ridge probes; error bars: seed SD\n'
        'Noise is added to representations; this is not a proof of exact information loss.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(root/'noise_sensitivity.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
