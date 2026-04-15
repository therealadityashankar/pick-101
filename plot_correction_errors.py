"""plot_correction_errors.py — Per-point errors after perspective correction.

Uses the linear-layer-derived parameters:
    Cx = -136.34, Cy = 139.91, Cz = 333.89  (global slope fit)
or per-layer mean:
    Cx = -145.82 (mean of 30/45/60), Cy = 141.73, Cz = 280.67

Shows errors with and without the 15mm layer, as histograms + bell curves.

Usage:
    uv run python plot_correction_errors.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ── Calibration parameters (from fit_linear_layers.py per-layer means, excl h=15) ──
# Using mean of 30/45/60mm layers which are consistent
Cx = np.mean([-145.37, -144.73, -145.35])   # = -145.15
Cy = np.mean([147.07, 137.76, 140.37])       # = 141.73
Cz = np.mean([(277.66+295.67)/2,             # mean of X and Y Cz per layer
               (282.95+259.01)/2,
               (279.38+264.98)/2])            # = 279.94


def correct(obs_x, obs_y, h, Cx, Cy, Cz):
    scale  = Cz / (Cz - h)
    true_x = Cx + (obs_x - Cx) / scale
    true_y = Cy + (obs_y - Cy) / scale
    return true_x, true_y


def compute_errors(true_x, true_y, obs_x, obs_y, heights, Cx, Cy, Cz):
    cx, cy = correct(obs_x, obs_y, heights, Cx, Cy, Cz)
    ex = cx - true_x
    ey = cy - true_y
    return ex, ey


def plot_bell(ax, data, label, color, bins=20):
    ax.hist(data, bins=bins, density=True, alpha=0.4, color=color, label=label)
    mu, sigma = norm.fit(data)
    xs = np.linspace(data.min() - 1, data.max() + 1, 200)
    ax.plot(xs, norm.pdf(xs, mu, sigma), color=color, lw=2,
            label=f'μ={mu:+.2f}  σ={sigma:.2f}mm')
    ax.axvline(mu, color=color, lw=1, ls='--', alpha=0.7)
    return mu, sigma


def print_table(label, true_x, true_y, obs_x, obs_y, heights, ex, ey):
    err_mag = np.sqrt(ex**2 + ey**2)
    print(f"\n{'─'*80}")
    print(f"{label}")
    print(f"{'#':>3}  {'H':>5}  {'TrX':>5}  {'TrY':>5}  "
          f"{'ObsX':>7}  {'ObsY':>7}  {'CorrX':>7}  {'CorrY':>7}  "
          f"{'ErrX':>7}  {'ErrY':>7}  {'|Err|':>7}")
    for i in range(len(true_x)):
        cx_i = true_x[i] + ex[i]
        cy_i = true_y[i] + ey[i]
        print(f"{i:3d}  {heights[i]:5.0f}  {true_x[i]:5.0f}  {true_y[i]:5.0f}  "
              f"{obs_x[i]:7.2f}  {obs_y[i]:7.2f}  {cx_i:7.2f}  {cy_i:7.2f}  "
              f"{ex[i]:+7.2f}  {ey[i]:+7.2f}  {err_mag[i]:7.3f}")
    print(f"\n  Mean X err: {ex.mean():+.3f}mm   Std: {ex.std():.3f}mm")
    print(f"  Mean Y err: {ey.mean():+.3f}mm   Std: {ey.std():.3f}mm")
    print(f"  Mean |err|: {err_mag.mean():.3f}mm  Std: {err_mag.std():.3f}mm")


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

    # Exclude h=0 from correction analysis (trivially near-zero error)
    nonzero = heights > 0
    mask_all  = nonzero
    mask_no15 = nonzero & (heights != 15)

    print(f"Calibration params (from 30/45/60mm layer means):")
    print(f"  Cx = {Cx:.2f} mm   Cy = {Cy:.2f} mm   Cz = {Cz:.2f} mm")

    datasets = [
        ("All non-zero layers (15+30+45+60mm)", mask_all,   'steelblue'),
        ("Without 15mm layer (30+45+60mm)",     mask_no15,  'tomato'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Per-point errors after correction\n"
        f"Cx={Cx:.1f}mm  Cy={Cy:.1f}mm  Cz={Cz:.1f}mm",
        fontsize=12)

    for col, (label, mask, color) in enumerate(datasets):
        ex, ey = compute_errors(
            true_x[mask], true_y[mask], obs_x[mask], obs_y[mask],
            heights[mask], Cx, Cy, Cz)

        print_table(label,
                    true_x[mask], true_y[mask], obs_x[mask], obs_y[mask],
                    heights[mask], ex, ey)

        # X error histogram + bell
        ax = axes[0, col]
        ax.set_title(f"X error — {label.split('(')[0].strip()}")
        plot_bell(ax, ex, 'X error', color)
        ax.set_xlabel("Corrected X − True X (mm)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Y error histogram + bell
        ax = axes[1, col]
        ax.set_title(f"Y error — {label.split('(')[0].strip()}")
        plot_bell(ax, ey, 'Y error', color)
        ax.set_xlabel("Corrected Y − True Y (mm)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.file).with_name("correction_errors_bell.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
