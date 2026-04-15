"""plot_jenga_calibration.py — Layer-wise X and Y deviation plots.

Usage:
    uv run python plot_jenga_calibration.py
    uv run python plot_jenga_calibration.py --file calibration_jenga.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LAYER_COLORS = {
    0:  '#444444',
    15: '#4a90d9',
    30: '#50c878',
    45: '#e0a030',
    60: '#e05050',
    75: '#9b59b6',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="calibration_jenga.npz")
    args = ap.parse_args()

    d       = np.load(args.file)
    true_x  = d['true_x'].astype(float)
    true_y  = d['true_y'].astype(float)
    obs_x   = d['obs_x'].astype(float)
    obs_y   = d['obs_y'].astype(float)
    heights = d['heights'].astype(float)

    err_x = obs_x - true_x
    err_y = obs_y - true_y

    layers = sorted(set(heights.tolist()))
    n      = len(layers)

    # ── Figure 1: X and Y deviations per layer, vs true X ────────────────────
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * n), sharex=False)
    fig.suptitle("Layer-wise deviations  (obs − true, interior mm)", fontsize=13, y=1.01)

    for row, h in enumerate(layers):
        mask   = heights == h
        col    = LAYER_COLORS.get(int(h), '#888888')
        tx, ty = true_x[mask], true_y[mask]
        ex, ey = err_x[mask], err_y[mask]

        for col_idx, (err, axis_label, true_pos) in enumerate([
            (ex, 'X error (mm)', tx),
            (ey, 'Y error (mm)', ty),
        ]):
            ax = axes[row, col_idx]

            # Scatter coloured by true Y (X panel) or true X (Y panel)
            other = ty if col_idx == 0 else tx
            sc = ax.scatter(true_pos, err, c=other, cmap='viridis',
                            s=55, zorder=4, edgecolors='white', linewidths=0.4)
            plt.colorbar(sc, ax=ax, pad=0.02,
                         label='True Y (mm)' if col_idx == 0 else 'True X (mm)')

            # Best-fit line
            if len(true_pos) > 1:
                p  = np.polyfit(true_pos, err, 1)
                xs = np.linspace(true_pos.min(), true_pos.max(), 100)
                ax.plot(xs, np.polyval(p, xs), color=col, lw=1.8, ls='--',
                        label=f'slope={p[0]:.3f}  intercept={p[1]:+.2f}')

            ax.axhline(err.mean(), color='red', lw=1, ls=':',
                       label=f'mean={err.mean():+.2f}mm')
            ax.axhline(0, color='black', lw=0.7)

            ax.set_title(f"h={h:.0f}mm — {'X' if col_idx==0 else 'Y'} error  "
                         f"(std={err.std():.2f}mm)", fontsize=9)
            ax.set_xlabel(f"True {'X' if col_idx==0 else 'Y'} (mm)", fontsize=8)
            ax.set_ylabel(axis_label, fontsize=8)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out1 = Path(args.file).with_name('jenga_deviations_per_layer.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved {out1}")
    plt.close()

    # ── Figure 2: All layers overlaid — X error and Y error vs true pos ───────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("All layers overlaid — error vs true position", fontsize=12)

    for h in layers:
        mask = heights == h
        col  = LAYER_COLORS.get(int(h), '#888888')
        lbl  = f"h={h:.0f}mm"
        ax1.scatter(true_x[mask], err_x[mask], color=col, s=40, alpha=0.8,
                    label=lbl, zorder=3)
        ax2.scatter(true_y[mask], err_y[mask], color=col, s=40, alpha=0.8,
                    label=lbl, zorder=3)
        # Fit line per layer
        if mask.sum() > 1:
            for ax, tx_m, err_m in [(ax1, true_x[mask], err_x[mask]),
                                     (ax2, true_y[mask], err_y[mask])]:
                p  = np.polyfit(tx_m, err_m, 1)
                xs = np.linspace(tx_m.min(), tx_m.max(), 100)
                ax.plot(xs, np.polyval(p, xs), color=col, lw=1.2, ls='--', alpha=0.7)

    for ax, xlabel in [(ax1, 'True X (mm)'), (ax2, 'True Y (mm)')]:
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax1.set_ylabel('X error: obs − true (mm)')
    ax1.set_title('X error vs True X')
    ax2.set_ylabel('Y error: obs − true (mm)')
    ax2.set_title('Y error vs True Y')

    plt.tight_layout()
    out2 = Path(args.file).with_name('jenga_deviations_overlaid.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved {out2}")
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'H(mm)':>6}  {'N':>3}  {'mean X err':>10}  {'std X':>6}  "
          f"{'mean Y err':>10}  {'std Y':>6}")
    for h in layers:
        mask = heights == h
        print(f"{h:6.0f}  {mask.sum():3d}  {err_x[mask].mean():+10.2f}  "
              f"{err_x[mask].std():6.2f}  {err_y[mask].mean():+10.2f}  "
              f"{err_y[mask].std():6.2f}")


if __name__ == "__main__":
    main()
