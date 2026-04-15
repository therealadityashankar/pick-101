"""plot_calibration_xy.py — X and Y error breakdown per cube with best-fit lines.

Usage:
    uv run python plot_calibration_xy.py
    uv run python plot_calibration_xy.py --file camera_calibration.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HEIGHT_COLORS = {20: '#e05050', 30: '#4a90d9', 50: '#50c878'}
CUBE_SIZES = [20, 30, 50]


def correct_observation(obs_x, obs_y, h, Cx, Cy, Cz):
    scale = Cz / (Cz - h)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="camera_calibration.npz")
    args = ap.parse_args()

    path = Path(args.file)
    d    = np.load(path)
    Cx, Cy, Cz = float(d['Cx']), float(d['Cy']), float(d['Cz'])
    inset_mm    = float(d['inset_mm'])
    meas        = d['measurements']   # (N,5): true_x, true_y, obs_x, obs_y, h

    true_x, true_y = meas[:, 0], meas[:, 1]
    obs_x,  obs_y  = meas[:, 2], meas[:, 3]
    heights        = meas[:, 4]

    err_x_raw = obs_x - true_x
    err_y_raw = obs_y - true_y

    corr_x = np.zeros_like(obs_x)
    corr_y = np.zeros_like(obs_y)
    for i, (ox, oy, h) in enumerate(zip(obs_x, obs_y, heights)):
        corr_x[i], corr_y[i] = correct_observation(ox, oy, h, Cx, Cy, Cz)
    err_x_corr = corr_x - true_x
    err_y_corr = corr_y - true_y

    # Distance from nadir along each axis
    dx_nadir = true_x - Cx   # signed X distance from nadir
    dy_nadir = true_y - Cy   # signed Y distance from nadir

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        f"X / Y error per cube  |  nadir=({Cx:.1f},{Cy:.1f}) mm  Cz={Cz:.1f} mm",
        fontsize=12)

    for row, cube_h in enumerate(CUBE_SIZES):
        mask  = heights == cube_h
        col   = HEIGHT_COLORS[cube_h]
        n     = mask.sum()

        ex_raw  = err_x_raw[mask]
        ey_raw  = err_y_raw[mask]
        ex_corr = err_x_corr[mask]
        ey_corr = err_y_corr[mask]
        dxn     = dx_nadir[mask]
        dyn     = dy_nadir[mask]
        idx     = np.arange(n)

        def fit_line(x, y):
            """Return (xs, ys_fit, slope, intercept) for a linear fit."""
            if len(x) < 2:
                return x, y, np.nan, np.nan
            p = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            return xs, np.polyval(p, xs), p[0], p[1]

        # ── Col 0: X error vs measurement index ──────────────────────────────
        ax = axes[row, 0]
        ax.bar(idx, ex_raw,  color=col, alpha=0.5, label='Raw')
        ax.bar(idx, ex_corr, color=col, alpha=1.0, label='Corrected',
               edgecolor='black', linewidth=0.7)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(ex_raw.mean(),  color='red',  lw=1, ls='--',
                   label=f'Mean raw {ex_raw.mean():+.1f}mm')
        ax.axhline(ex_corr.mean(), color='green', lw=1, ls='--',
                   label=f'Mean corr {ex_corr.mean():+.1f}mm')
        ax.set_title(f"{cube_h}mm — X error per point")
        ax.set_xlabel("Measurement index"); ax.set_ylabel("X error (mm)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

        # ── Col 1: Y error vs measurement index ──────────────────────────────
        ax = axes[row, 1]
        ax.bar(idx, ey_raw,  color=col, alpha=0.5, label='Raw')
        ax.bar(idx, ey_corr, color=col, alpha=1.0, label='Corrected',
               edgecolor='black', linewidth=0.7)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(ey_raw.mean(),  color='red',   lw=1, ls='--',
                   label=f'Mean raw {ey_raw.mean():+.1f}mm')
        ax.axhline(ey_corr.mean(), color='green', lw=1, ls='--',
                   label=f'Mean corr {ey_corr.mean():+.1f}mm')
        ax.set_title(f"{cube_h}mm — Y error per point")
        ax.set_xlabel("Measurement index"); ax.set_ylabel("Y error (mm)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

        # ── Col 2: X error vs X distance from nadir ──────────────────────────
        ax = axes[row, 2]
        ax.scatter(dxn, ex_raw,  color=col, s=60, alpha=0.6, label='Raw',       zorder=3)
        ax.scatter(dxn, ex_corr, color=col, s=60, alpha=1.0, label='Corrected',
                   edgecolors='black', linewidths=0.8, zorder=4)
        xs, ys, slope, intc = fit_line(dxn, ex_raw)
        ax.plot(xs, ys, color='red', lw=1.5, ls='--',
                label=f'Raw fit  slope={slope:.3f}')
        xs2, ys2, slope2, intc2 = fit_line(dxn, ex_corr)
        ax.plot(xs2, ys2, color='green', lw=1.5, ls='--',
                label=f'Corr fit slope={slope2:.3f}')
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8, ls=':')
        ax.set_title(f"{cube_h}mm — X error vs ΔX from nadir")
        ax.set_xlabel("X distance from nadir (mm)"); ax.set_ylabel("X error (mm)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # ── Col 3: Y error vs Y distance from nadir ──────────────────────────
        ax = axes[row, 3]
        ax.scatter(dyn, ey_raw,  color=col, s=60, alpha=0.6, label='Raw',       zorder=3)
        ax.scatter(dyn, ey_corr, color=col, s=60, alpha=1.0, label='Corrected',
                   edgecolors='black', linewidths=0.8, zorder=4)
        xs, ys, slope, intc = fit_line(dyn, ey_raw)
        ax.plot(xs, ys, color='red', lw=1.5, ls='--',
                label=f'Raw fit  slope={slope:.3f}')
        xs2, ys2, slope2, intc2 = fit_line(dyn, ey_corr)
        ax.plot(xs2, ys2, color='green', lw=1.5, ls='--',
                label=f'Corr fit slope={slope2:.3f}')
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8, ls=':')
        ax.set_title(f"{cube_h}mm — Y error vs ΔY from nadir")
        ax.set_xlabel("Y distance from nadir (mm)"); ax.set_ylabel("Y error (mm)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.with_name(path.stem + '_xy_errors.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
