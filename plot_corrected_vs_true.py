"""plot_corrected_vs_true.py — Layer-wise corrected vs true position plots.

Usage:
    uv run python plot_corrected_vs_true.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

Cx = -145.15
Cy = 141.73
Cz = 276.61

LAYER_COLORS = {0: '#888888', 15: '#4a90d9', 30: '#50c878', 45: '#e0a030', 60: '#e05050'}


def correct(obs_x, obs_y, h):
    scale = Cz / (Cz - h)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


def main():
    d       = np.load("calibration_jenga.npz")
    true_x  = d['true_x'].astype(float)
    true_y  = d['true_y'].astype(float)
    obs_x   = d['obs_x'].astype(float)
    obs_y   = d['obs_y'].astype(float)
    heights = d['heights'].astype(float)

    layers = [h for h in sorted(set(heights.tolist())) if h > 0]
    n = len(layers)

    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    fig.suptitle(
        f"Corrected vs True position — per layer\n"
        f"Cx={Cx:.1f}  Cy={Cy:.1f}  Cz={Cz:.1f} mm",
        fontsize=12)

    all_vals = np.concatenate([true_x, true_y])
    lim = (all_vals.min() - 5, all_vals.max() + 5)
    ideal = np.array(lim)

    for row, h in enumerate(layers):
        mask = heights == h
        col  = LAYER_COLORS.get(int(h), '#888')
        cx, cy = correct(obs_x[mask], obs_y[mask], h)
        tx, ty = true_x[mask], true_y[mask]

        for col_idx, (true_vals, corr_vals, axis) in enumerate([
            (tx, cx, 'X'),
            (ty, cy, 'Y'),
        ]):
            ax = axes[row, col_idx]
            err = corr_vals - true_vals

            ax.scatter(true_vals, corr_vals, color=col, s=60, zorder=4,
                       edgecolors='white', linewidths=0.5)

            # Annotate each point with its error
            for tv, cv in zip(true_vals, corr_vals):
                ax.annotate(f'{cv-tv:+.2f}', (tv, cv),
                            textcoords='offset points', xytext=(5, 3),
                            fontsize=6.5, color='#333333')

            ax.plot(ideal, ideal, 'k--', lw=1, alpha=0.5, label='Ideal')
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.set_title(f"h={h:.0f}mm — {axis}   "
                         f"μ={err.mean():+.2f}mm  σ={err.std():.2f}mm",
                         fontsize=9)
            ax.set_xlabel(f"True {axis} (mm)")
            ax.set_ylabel(f"Corrected {axis} (mm)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path("jenga_corrected_vs_true.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
