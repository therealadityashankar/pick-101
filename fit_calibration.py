"""fit_calibration.py — Fit the perspective correction model to jenga calibration data.

Model
-----
The camera is at position (Cx, Cy) in board interior mm and height Cz above it.
An object at true position (tx, ty) and height h above the board appears at:

    obs_x = Cx + (tx - Cx) * Cz / (Cz - h)
    obs_y = Cy + (ty - Cy) * Cz / (Cz - h)

Rearranged as error:
    err_x = obs_x - tx = (tx - Cx) * h / (Cz - h)
    err_y = obs_y - ty = (ty - Cy) * h / (Cz - h)

We fit (Cx, Cy, Cz) via least squares over all measurements.

Usage:
    uv run python fit_calibration.py
    uv run python fit_calibration.py --file calibration_jenga.npz --outlier-threshold 5.0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def residuals(params, true_x, true_y, obs_x, obs_y, heights):
    Cx, Cy, Cz = params
    scale = Cz / (Cz - heights)
    pred_x = Cx + (true_x - Cx) * scale
    pred_y = Cy + (true_y - Cy) * scale
    return np.concatenate([pred_x - obs_x, pred_y - obs_y])


def fit(true_x, true_y, obs_x, obs_y, heights):
    x0 = [40.0, 40.0, 300.0]
    result = least_squares(
        residuals, x0,
        args=(true_x, true_y, obs_x, obs_y, heights),
        bounds=([0, 0, 50], [200, 200, 2000]),
    )
    return result.x, result


def correction(obs_x, obs_y, h, Cx, Cy, Cz):
    scale = Cz / (Cz - h)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",               default="calibration_jenga.npz")
    ap.add_argument("--outlier-threshold",  type=float, default=5.0,
                    help="Residual magnitude (mm) above which a point is flagged as outlier")
    args = ap.parse_args()

    d       = np.load(args.file)
    true_x  = d['true_x'].astype(float)
    true_y  = d['true_y'].astype(float)
    obs_x   = d['obs_x'].astype(float)
    obs_y   = d['obs_y'].astype(float)
    heights = d['heights'].astype(float)
    N       = len(true_x)

    # ── Fit all data ──────────────────────────────────────────────────────────
    (Cx, Cy, Cz), res_all = fit(true_x, true_y, obs_x, obs_y, heights)

    # Per-point residual magnitude after correction
    cx_all, cy_all = correction(obs_x, obs_y, heights, Cx, Cy, Cz)
    resid = np.sqrt((cx_all - true_x)**2 + (cy_all - true_y)**2)

    print("=" * 60)
    print(f"FIT (all {N} points)")
    print(f"  Cx = {Cx:.2f} mm   Cy = {Cy:.2f} mm   Cz = {Cz:.2f} mm")
    print(f"  Mean correction residual : {resid.mean():.3f} mm")
    print(f"  Std  correction residual : {resid.std():.3f} mm")
    print(f"  Max  correction residual : {resid.max():.3f} mm")
    print()

    # ── Identify outliers ─────────────────────────────────────────────────────
    outlier_mask = resid > args.outlier_threshold
    n_out = outlier_mask.sum()
    print(f"Outliers (residual > {args.outlier_threshold} mm): {n_out}/{N}")
    print(f"  {'#':>3}  {'H':>5}  {'TrueX':>6}  {'TrueY':>6}  "
          f"{'ObsX':>7}  {'ObsY':>7}  {'ResidX':>7}  {'ResidY':>7}  {'|Resid|':>7}")
    for i in np.where(outlier_mask)[0]:
        rx = cx_all[i] - true_x[i]
        ry = cy_all[i] - true_y[i]
        print(f"  {i:3d}  {heights[i]:5.0f}  {true_x[i]:6.0f}  {true_y[i]:6.0f}  "
              f"{obs_x[i]:7.2f}  {obs_y[i]:7.2f}  {rx:+7.2f}  {ry:+7.2f}  {resid[i]:7.3f}")
    print()

    # ── Fit without outliers ──────────────────────────────────────────────────
    good = ~outlier_mask
    (Cx2, Cy2, Cz2), res_good = fit(
        true_x[good], true_y[good], obs_x[good], obs_y[good], heights[good])
    cx_g, cy_g = correction(obs_x[good], obs_y[good], heights[good], Cx2, Cy2, Cz2)
    resid2 = np.sqrt((cx_g - true_x[good])**2 + (cy_g - true_y[good])**2)

    print(f"FIT (without {n_out} outliers — {good.sum()} points)")
    print(f"  Cx = {Cx2:.2f} mm   Cy = {Cy2:.2f} mm   Cz = {Cz2:.2f} mm")
    print(f"  Mean correction residual : {resid2.mean():.3f} mm")
    print(f"  Std  correction residual : {resid2.std():.3f} mm")
    print(f"  Max  correction residual : {resid2.max():.3f} mm")
    print()

    # Save clean calibration
    out_npz = Path(args.file).with_name("camera_calibration_jenga.npz")
    np.savez(out_npz, Cx=Cx2, Cy=Cy2, Cz=Cz2,
             inset_mm=d['inset_mm'],
             measurements=np.stack([true_x[good], true_y[good],
                                     obs_x[good], obs_y[good],
                                     heights[good]], axis=1))
    print(f"Clean calibration saved to {out_npz}")
    print(f"\nCorrection formula:")
    print(f"  scale  = {Cz2:.1f} / ({Cz2:.1f} - h)")
    print(f"  true_x = {Cx2:.2f} + (obs_x - {Cx2:.2f}) / scale")
    print(f"  true_y = {Cy2:.2f} + (obs_y - {Cy2:.2f}) / scale")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Perspective model fit\n"
        f"All: Cx={Cx:.1f} Cy={Cy:.1f} Cz={Cz:.1f}mm  |  "
        f"Clean: Cx={Cx2:.1f} Cy={Cy2:.1f} Cz={Cz2:.1f}mm",
        fontsize=10)

    layers = sorted(set(heights.tolist()))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))

    # Panel 1: raw error vs height (shows model fit)
    ax = axes[0]
    ax.set_title("Mean |error| vs height")
    raw_err = np.sqrt((obs_x - true_x)**2 + (obs_y - true_y)**2)
    for li, h in enumerate(layers):
        m = heights == h
        ax.scatter([h]*m.sum(), raw_err[m], color=colors[li], s=30, alpha=0.6)
        ax.scatter(h, raw_err[m].mean(), color=colors[li], s=120, marker='D',
                   zorder=5, label=f"h={h:.0f}mm  mean={raw_err[m].mean():.1f}")
    ax.set_xlabel("Height (mm)"); ax.set_ylabel("|error| mm (raw)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: corrected residual (all)
    ax = axes[1]
    ax.set_title(f"Corrected residual — all {N} pts")
    for li, h in enumerate(layers):
        m = heights == h
        ax.scatter(true_x[m], resid[m], color=colors[li], s=35, alpha=0.8,
                   label=f"h={h:.0f}mm")
    ax.scatter(true_x[outlier_mask], resid[outlier_mask],
               s=120, marker='x', color='red', linewidths=2, zorder=6,
               label=f'Outliers (>{args.outlier_threshold}mm)')
    ax.axhline(args.outlier_threshold, color='red', lw=1, ls='--')
    ax.axhline(resid.mean(), color='black', lw=1, ls=':', label=f'mean={resid.mean():.2f}mm')
    ax.set_xlabel("True X (mm)"); ax.set_ylabel("Residual after correction (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 3: corrected residual (clean)
    ax = axes[2]
    ax.set_title(f"Corrected residual — clean {good.sum()} pts")
    for li, h in enumerate(layers):
        m = heights[good] == h
        ax.scatter(true_x[good][m], resid2[m], color=colors[li], s=35, alpha=0.8,
                   label=f"h={h:.0f}mm  μ={resid2[m].mean():.2f}")
    ax.axhline(resid2.mean(), color='black', lw=1, ls=':', label=f'mean={resid2.mean():.2f}mm')
    ax.set_xlabel("True X (mm)"); ax.set_ylabel("Residual after correction (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = Path(args.file).with_name("jenga_fit.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_png}")


if __name__ == "__main__":
    main()
