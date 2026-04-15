"""plot_x_calibration.py — Graph the X-axis calibration data from calibration_x_raw.npz.

Shows:
  1. Observed vs True X  (with ideal 1:1 line and best-fit line)
  2. X error (obs - true) vs True X  (with best-fit line)

Usage:
    uv run python plot_x_calibration.py
    uv run python plot_x_calibration.py --file calibration_x_raw.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="calibration_x_raw.npz")
    args = ap.parse_args()

    d      = np.load(args.file)
    true_x = d['true_x'].astype(float)
    obs_x  = d['obs_x'].astype(float)
    err_x  = obs_x - true_x

    # Best-fit lines
    p_obs = np.polyfit(true_x, obs_x, 1)
    p_err = np.polyfit(true_x, err_x, 1)
    xs    = np.linspace(true_x.min() - 2, true_x.max() + 2, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("X-axis calibration: bordered ArUco TL corner\n"
                 f"(n={len(true_x)} points, interior mm)",
                 fontsize=12)

    # ── Left: observed vs true ────────────────────────────────────────────────
    ax1.scatter(true_x, obs_x, color='steelblue', s=80, zorder=4, label='Measurements')
    ax1.plot(xs, np.polyval(p_obs, xs), 'steelblue', lw=1.5, ls='--',
             label=f'Fit:  obs = {p_obs[0]:.3f}·true + {p_obs[1]:.2f}')
    ax1.plot(xs, xs, 'k', lw=1, ls=':', label='Ideal (obs = true)')

    # Annotate each point with its index
    for i, (tx, ox) in enumerate(zip(true_x, obs_x)):
        ax1.annotate(f'{i+1}', (tx, ox), textcoords='offset points',
                     xytext=(6, 3), fontsize=8, color='steelblue')

    ax1.set_xlabel("True X — TL corner (interior mm)")
    ax1.set_ylabel("Observed X — TL corner (interior mm)")
    ax1.set_title("Observed vs True X")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ── Right: error vs true ──────────────────────────────────────────────────
    ax2.scatter(true_x, err_x, color='tomato', s=80, zorder=4, label='X error')
    ax2.plot(xs, np.polyval(p_err, xs), 'tomato', lw=1.5, ls='--',
             label=f'Fit:  slope={p_err[0]:.4f}  intercept={p_err[1]:.2f}')
    ax2.axhline(err_x.mean(), color='darkred', lw=1, ls='-.',
                label=f'Mean error = {err_x.mean():+.2f} mm')
    ax2.axhline(0, color='black', lw=0.8)

    for i, (tx, ex) in enumerate(zip(true_x, err_x)):
        ax2.annotate(f'{i+1}', (tx, ex), textcoords='offset points',
                     xytext=(6, 3), fontsize=8, color='tomato')

    ax2.set_xlabel("True X — TL corner (interior mm)")
    ax2.set_ylabel("X error: obs − true (mm)")
    ax2.set_title("X Error vs True X")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.file).with_name('calibration_x_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")

    print(f"\nSummary:")
    print(f"  Mean X error : {err_x.mean():+.2f} mm")
    print(f"  Std X error  : {err_x.std():.2f} mm")
    print(f"  Fit slope    : {p_obs[0]:.4f}  (ideal = 1.0)")
    print(f"  Fit intercept: {p_obs[1]:.2f} mm")


if __name__ == "__main__":
    main()
