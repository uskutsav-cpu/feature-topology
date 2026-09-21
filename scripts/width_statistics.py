"""Deterministic visualization of the frozen finite-width terminology gate."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['svg.hashsalt']='feature-topology-v3'


def plot_width_gate(gate: dict, output: str | Path) -> None:
    if (gate.get('schema')!='feature-topology.width-scaling-result.v1'
            or gate.get('status')!='complete'):
        raise ValueError('Finite-width gate is incomplete')
    fits=sorted(gate.get('fits',[]),key=lambda row:row['width'])
    if not fits or any(row.get('status')!='ok' for row in fits):
        raise ValueError('Finite-width fits are incomplete')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    widths=[row['width'] for row in fits]
    centers=[row['transition']['center_gamma'] for row in fits]
    transition_widths=[row['transition']['width_10_90_log2_gamma'] for row in fits]
    aicc_deltas=[row['delta_aicc_smooth_minus_transition'] for row in fits]
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    axes[0].plot(widths,centers,marker='o')
    axes[0].set_ylabel('Fitted center γ')
    axes[1].plot(widths,transition_widths,marker='o')
    axes[1].set_ylabel('10–90% width (log₂ γ)')
    axes[2].plot(widths,aicc_deltas,marker='o')
    axes[2].axhline(0,color='black',linewidth=.8,alpha=.5)
    axes[2].set_ylabel('ΔAICc (smooth − transition)')
    for ax in axes:
        ax.set_xscale('log',base=2)
        ax.set_xlabel('Network width')
        ax.grid(alpha=.2)
    allowed=gate.get('phase_transition_language_allowed') is True
    fig.suptitle(f"Finite-width gate: {gate.get('terminology')} "
                 f"(phase-transition language {'allowed' if allowed else 'not allowed'})")
    for suffix in ['png','pdf','svg']:
        metadata=({'CreationDate':None,'ModDate':None} if suffix=='pdf'
                  else {'Date':None} if suffix=='svg' else None)
        fig.savefig(output/f'finite_width_gate.{suffix}',dpi=220,metadata=metadata)
    plt.close(fig)
