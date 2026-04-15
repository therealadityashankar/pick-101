"""plot_calibration.py — Visualise camera_calibration.npz errors.

Shows:
  1. Scatter: observed vs true positions (raw + corrected), coloured by height
  2. Error vector field: arrows from true → observed (raw) across the board
  3. Residual bar chart per measurement, grouped by cube height
  4. Error vs distance from camera nadir (should be linear if model is correct)

Usage:
    uv run python plot_calibration.py
    uv run python plot_calibration.py --file camera_calibration.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HEIGHT_COLORS = {20: '#e05050', 30: '#50a0e0', 50: '#50c878'}


def correct_observation(obs_x, obs_y, h, Cx, Cy, Cz):
    scale = Cz / (Cz - h)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="camera_calibration.npz")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    d = np.load(path)
    Cx, Cy, Cz = float(d['Cx']), float(d['Cy']), float(d['Cz'])
    inset_mm    = float(d['inset_mm'])
    square_mm   = float(d['square_mm'])

    if 'measurements' not in d:
        print("No measurements saved in this file. Re-run calibrate_3d.py to regenerate.")
        return

    meas = d['measurements']   # (N, 5): true_x, true_y, obs_x, obs_y, h
    true_x, true_y = meas[:, 0], meas[:, 1]
    obs_x,  obs_y  = meas[:, 2], meas[:, 3]
    heights        = meas[:, 4]

    # Corrected positions
    corr_x = np.zeros_like(obs_x)
    corr_y = np.zeros_like(obs_y)
    for i, (ox, oy, h) in enumerate(zip(obs_x, obs_y, heights)):
        corr_x[i], corr_y[i] = correct_observation(ox, oy, h, Cx, Cy, Cz)

    raw_err  = np.sqrt((obs_x  - true_x)**2 + (obs_y  - true_y)**2)
    corr_err = np.sqrt((corr_x - true_x)**2 + (corr_y - true_y)**2)

    # Distance from camera nadir in board mm
    nadir_dist = np.sqrt((true_x - Cx)**2 + (true_y - Cy)**2)

    unique_h = sorted(set(heights))
    colors   = [HEIGHT_COLORS.get(int(h), '#aaaaaa') for h in heights]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Calibration results  |  Cam nadir=({Cx:.1f},{Cy:.1f}) mm  height={Cz:.1f} mm  "
        f"|  mean raw={raw_err.mean():.2f}mm  mean corrected={corr_err.mean():.2f}mm",
        fontsize=11)

    # ── 1. Board scatter ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("Board positions (interior coords)")
    ax1.set_aspect('equal')
    isize = square_mm - 2 * inset_mm
    ax1.set_xlim(-5, isize + 5)
    ax1.set_ylim(isize + 5, -5)   # Y down to match board convention
    ax1.add_patch(plt.Rectangle((0, 0), isize, isize,
                                fill=False, edgecolor='black', lw=1.5))
    ax1.axhline(0, color='black', lw=0.5, ls='--')
    ax1.axvline(0, color='black', lw=0.5, ls='--')

    for i, (tx, ty, ox, oy, cx, cy, h) in enumerate(
            zip(true_x, true_y, obs_x, obs_y, corr_x, corr_y, heights)):
        col = HEIGHT_COLORS.get(int(h), '#aaaaaa')
        ix_t, iy_t = tx - inset_mm, ty - inset_mm
        ix_o, iy_o = ox - inset_mm, oy - inset_mm
        ix_c, iy_c = cx - inset_mm, cy - inset_mm
        ax1.plot(ix_t, iy_t, 'k+', ms=8, mew=1.5)
        ax1.plot(ix_o, iy_o, 'o', color=col, ms=6, alpha=0.7)
        ax1.plot(ix_c, iy_c, 's', color=col, ms=5, alpha=0.9,
                 markerfacecolor='none', markeredgewidth=1.5)
        ax1.annotate("", xy=(ix_o, iy_o), xytext=(ix_t, iy_t),
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

    # Camera nadir
    ax1.plot(Cx - inset_mm, Cy - inset_mm, '*', color='gold',
             ms=14, zorder=5, label='Cam nadir')

    legend_handles = [
        mpatches.Patch(color=HEIGHT_COLORS.get(int(h), '#aaa'), label=f'{int(h)}mm cube')
        for h in unique_h
    ] + [
        plt.Line2D([0],[0], marker='+', color='black', ls='none', ms=8, label='True'),
        plt.Line2D([0],[0], marker='o', color='grey',  ls='none', ms=6, label='Observed'),
        plt.Line2D([0],[0], marker='s', color='grey',  ls='none', ms=5,
                   markerfacecolor='none', label='Corrected'),
        plt.Line2D([0],[0], marker='*', color='gold',  ls='none', ms=10, label='Cam nadir'),
    ]
    ax1.legend(handles=legend_handles, fontsize=7, loc='lower right')
    ax1.set_xlabel("Interior X (mm)"); ax1.set_ylabel("Interior Y (mm)")
    ax1.grid(True, alpha=0.3)

    # ── 2. Error vector field ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("Error vectors: true → observed")
    ax2.set_aspect('equal')
    ax2.set_xlim(-5, isize + 5); ax2.set_ylim(isize + 5, -5)
    ax2.add_patch(plt.Rectangle((0, 0), isize, isize,
                                fill=False, edgecolor='black', lw=1.5))
    scale_factor = 5.0  # exaggerate arrows
    for tx, ty, ox, oy, h in zip(true_x, true_y, obs_x, obs_y, heights):
        col = HEIGHT_COLORS.get(int(h), '#aaaaaa')
        ix_t, iy_t = tx - inset_mm, ty - inset_mm
        dx, dy = (ox - tx) * scale_factor, (oy - ty) * scale_factor
        ax2.annotate("", xy=(ix_t + dx, iy_t + dy), xytext=(ix_t, iy_t),
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    ax2.plot(Cx - inset_mm, Cy - inset_mm, '*', color='gold', ms=14, zorder=5)
    ax2.set_xlabel("Interior X (mm)"); ax2.set_ylabel("Interior Y (mm)")
    ax2.set_title(f"Error vectors (×{scale_factor:.0f} scale)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Bar chart: raw vs corrected per measurement ────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("Error per measurement")
    n = len(meas)
    idx = np.arange(n)
    ax3.bar(idx - 0.2, raw_err,  0.35, label='Raw',       color=colors, alpha=0.6)
    ax3.bar(idx + 0.2, corr_err, 0.35, label='Corrected', color=colors, alpha=1.0,
            edgecolor='black', linewidth=0.5)
    ax3.axhline(raw_err.mean(),  color='red',   ls='--', lw=1, label=f'Mean raw {raw_err.mean():.2f}mm')
    ax3.axhline(corr_err.mean(), color='green', ls='--', lw=1, label=f'Mean corr {corr_err.mean():.2f}mm')
    ax3.set_xlabel("Measurement index"); ax3.set_ylabel("Error (mm)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')

    # ── 4. Error vs nadir distance ────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("Raw error vs distance from cam nadir")
    for h in unique_h:
        mask = heights == h
        col  = HEIGHT_COLORS.get(int(h), '#aaa')
        ax4.scatter(nadir_dist[mask], raw_err[mask], color=col,
                    label=f'{int(h)}mm', s=60, zorder=3)
    # Fit line
    if len(nadir_dist) > 1:
        p = np.polyfit(nadir_dist, raw_err, 1)
        xs = np.linspace(nadir_dist.min(), nadir_dist.max(), 100)
        ax4.plot(xs, np.polyval(p, xs), 'k--', lw=1, label=f'fit slope={p[0]:.3f}')
    ax4.set_xlabel("Distance from nadir (mm)"); ax4.set_ylabel("Raw error (mm)")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    # ── 5. Error vs height ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("Error vs cube height")
    for h in unique_h:
        mask = heights == h
        col  = HEIGHT_COLORS.get(int(h), '#aaa')
        ax5.scatter([h]*mask.sum(), raw_err[mask],  marker='o', color=col,
                    s=60, alpha=0.7, label=f'{int(h)}mm raw')
        ax5.scatter([h]*mask.sum(), corr_err[mask], marker='s', color=col,
                    s=60, alpha=1.0, edgecolors='black', linewidths=0.8)
    ax5.set_xlabel("Cube height (mm)"); ax5.set_ylabel("Error (mm)")
    ax5.set_xticks([int(h) for h in unique_h])
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

    # ── 6. X and Y error components ──────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("X / Y error components (raw)")
    err_x_raw = obs_x - true_x
    err_y_raw = obs_y - true_y
    for h in unique_h:
        mask = heights == h
        col  = HEIGHT_COLORS.get(int(h), '#aaa')
        ax6.scatter(err_x_raw[mask], err_y_raw[mask], color=col,
                    label=f'{int(h)}mm', s=60, zorder=3)
    ax6.axhline(0, color='black', lw=0.8)
    ax6.axvline(0, color='black', lw=0.8)
    ax6.set_xlabel("X error (mm)"); ax6.set_ylabel("Y error (mm)")
    ax6.set_aspect('equal')
    ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = path.with_suffix('.png').with_name(path.stem + '_plot.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
