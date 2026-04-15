"""fit_linear_layers.py — Fit a linear equation per layer, then find Cx/Cy/Cz
from the relationship between slope/intercept and height.

Per-layer model:
    err_x = obs_x - true_x = slope_x * true_x + intercept_x
    err_y = obs_y - true_y = slope_y * true_y + intercept_y

From the perspective model:
    slope_x     = h / (Cz - h)          → Cz = h * (1 + 1/slope_x)
    intercept_x = -Cx * slope_x         → Cx = -intercept_x / slope_x
    (same for Y → Cy)

Usage:
    uv run python fit_linear_layers.py
    uv run python fit_linear_layers.py --file calibration_jenga.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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
    # Skip h=0 for slope fitting (slope=0, no information about Cz)
    layers_nonzero = [h for h in layers if h > 0]

    # ── Per-layer linear fits ─────────────────────────────────────────────────
    print(f"{'H(mm)':>6}  {'slope_x':>9}  {'intercept_x':>12}  "
          f"{'slope_y':>9}  {'intercept_y':>12}  {'N':>3}")

    slopes_x, intercepts_x = [], []
    slopes_y, intercepts_y = [], []
    Czs_x, Cxs, Czs_y, Cys = [], [], [], []

    for h in layers_nonzero:
        mask = heights == h
        tx, ty = true_x[mask], true_y[mask]
        ex, ey = err_x[mask], err_y[mask]

        px = np.polyfit(tx, ex, 1)
        py = np.polyfit(ty, ey, 1)

        sx, bx = px[0], px[1]
        sy, by = py[0], py[1]

        slopes_x.append(sx);     intercepts_x.append(bx)
        slopes_y.append(sy);     intercepts_y.append(by)

        # Recover Cz and Cx from this layer
        Cz_from_x = h * (1 + 1/sx) if sx != 0 else np.nan
        Cx_from_x = -bx / sx      if sx != 0 else np.nan
        Cz_from_y = h * (1 + 1/sy) if sy != 0 else np.nan
        Cy_from_y = -by / sy      if sy != 0 else np.nan

        Czs_x.append(Cz_from_x); Cxs.append(Cx_from_x)
        Czs_y.append(Cz_from_y); Cys.append(Cy_from_y)

        print(f"{h:6.0f}  {sx:9.4f}  {bx:12.4f}  {sy:9.4f}  {by:12.4f}  {mask.sum():3d}")

    slopes_x    = np.array(slopes_x)
    intercepts_x= np.array(intercepts_x)
    slopes_y    = np.array(slopes_y)
    intercepts_y= np.array(intercepts_y)
    hs          = np.array(layers_nonzero)

    print()
    print("Per-layer estimates of camera position:")
    print(f"{'H(mm)':>6}  {'Cx (from X)':>12}  {'Cz (from X)':>12}  "
          f"{'Cy (from Y)':>12}  {'Cz (from Y)':>12}")
    for i, h in enumerate(layers_nonzero):
        print(f"{h:6.0f}  {Cxs[i]:12.2f}  {Czs_x[i]:12.2f}  "
              f"{Cys[i]:12.2f}  {Czs_y[i]:12.2f}")

    print()
    print(f"Mean Cx = {np.nanmean(Cxs):+.2f} mm  (std {np.nanstd(Cxs):.2f})")
    print(f"Mean Cy = {np.nanmean(Cys):+.2f} mm  (std {np.nanstd(Cys):.2f})")
    print(f"Mean Cz (X) = {np.nanmean(Czs_x):.2f} mm  (std {np.nanstd(Czs_x):.2f})")
    print(f"Mean Cz (Y) = {np.nanmean(Czs_y):.2f} mm  (std {np.nanstd(Czs_y):.2f})")

    # ── Fit slope vs h: slope = h/(Cz-h)  →  1/slope = Cz/h - 1 ─────────────
    # Linearise: 1/slope = (Cz)*1/h - 1  →  linear in 1/h
    inv_h  = 1.0 / hs
    px_cz  = np.polyfit(inv_h, 1.0/slopes_x, 1)  # slope=Cz, intercept=-1
    py_cz  = np.polyfit(inv_h, 1.0/slopes_y, 1)
    Cz_fit_x = px_cz[0]
    Cz_fit_y = py_cz[0]

    # intercept vs slope: intercept = -Cx * slope  →  linear through origin
    px_cx = np.polyfit(slopes_x, intercepts_x, 1)  # slope = -Cx
    py_cy = np.polyfit(slopes_y, intercepts_y, 1)
    Cx_fit = -px_cx[0]
    Cy_fit = -py_cy[0]

    print()
    print("Global fit from slope/intercept relationships:")
    print(f"  Cx = {Cx_fit:.2f} mm")
    print(f"  Cy = {Cy_fit:.2f} mm")
    print(f"  Cz (from X slopes) = {Cz_fit_x:.2f} mm")
    print(f"  Cz (from Y slopes) = {Cz_fit_y:.2f} mm")
    Cz_final = (Cz_fit_x + Cz_fit_y) / 2
    print(f"  Cz (averaged)      = {Cz_final:.2f} mm")

    print()
    print("Correction formula:")
    print(f"  scale  = {Cz_final:.1f} / ({Cz_final:.1f} - h)")
    print(f"  true_x = {Cx_fit:.2f} + (obs_x - {Cx_fit:.2f}) / scale")
    print(f"  true_y = {Cy_fit:.2f} + (obs_y - {Cy_fit:.2f}) / scale")

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    color_map = {h: colors[i] for i, h in enumerate(layers)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Linear layer fits  →  Cx={Cx_fit:.1f}mm  Cy={Cy_fit:.1f}mm  Cz={Cz_final:.1f}mm",
        fontsize=12)

    xs_range = np.linspace(true_x.min()-5, true_x.max()+5, 100)
    ys_range = np.linspace(true_y.min()-5, true_y.max()+5, 100)

    # Row 0: X error per layer with fit lines
    ax = axes[0, 0]
    ax.set_title("X error vs True X — per layer")
    for i, h in enumerate(layers):
        mask = heights == h
        col  = color_map[h]
        ax.scatter(true_x[mask], err_x[mask], color=col, s=40, alpha=0.8,
                   label=f"h={h:.0f}mm", zorder=3)
        if h > 0:
            j = layers_nonzero.index(h)
            ax.plot(xs_range, np.polyval([slopes_x[j], intercepts_x[j]], xs_range),
                    color=col, lw=1.5, ls='--')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("True X (mm)"); ax.set_ylabel("X error (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 1: Y error per layer with fit lines
    ax = axes[1, 0]
    ax.set_title("Y error vs True Y — per layer")
    for i, h in enumerate(layers):
        mask = heights == h
        col  = color_map[h]
        ax.scatter(true_y[mask], err_y[mask], color=col, s=40, alpha=0.8,
                   label=f"h={h:.0f}mm", zorder=3)
        if h > 0:
            j = layers_nonzero.index(h)
            ax.plot(ys_range, np.polyval([slopes_y[j], intercepts_y[j]], ys_range),
                    color=col, lw=1.5, ls='--')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("True Y (mm)"); ax.set_ylabel("Y error (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Slope vs h
    ax = axes[0, 1]
    ax.set_title("Slope vs Height")
    ax.scatter(hs, slopes_x, color='steelblue', s=80, label='X slope', zorder=3)
    ax.scatter(hs, slopes_y, color='tomato',    s=80, label='Y slope', zorder=3)
    h_range = np.linspace(0, hs.max()+5, 100)
    ax.plot(h_range, h_range / (Cz_final - h_range), 'steelblue', lw=1.5, ls='--',
            label=f'model h/(Cz-h), Cz={Cz_final:.0f}')
    ax.set_xlabel("Height h (mm)"); ax.set_ylabel("Slope")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Intercept vs slope (should be linear through origin, slope=-Cx or -Cy)
    ax = axes[1, 1]
    ax.set_title("Intercept vs Slope (→ nadir position)")
    ax.scatter(slopes_x, intercepts_x, color='steelblue', s=80, label='X', zorder=3)
    ax.scatter(slopes_y, intercepts_y, color='tomato',    s=80, label='Y', zorder=3)
    sl_range = np.linspace(0, max(slopes_x.max(), slopes_y.max())*1.1, 100)
    ax.plot(sl_range, np.polyval(px_cx, sl_range), 'steelblue', lw=1.5, ls='--',
            label=f'X fit  Cx={Cx_fit:.1f}mm')
    ax.plot(sl_range, np.polyval(py_cy, sl_range), 'tomato',    lw=1.5, ls='--',
            label=f'Y fit  Cy={Cy_fit:.1f}mm')
    ax.set_xlabel("Slope"); ax.set_ylabel("Intercept (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Per-layer Cz estimates
    ax = axes[0, 2]
    ax.set_title("Cz estimate per layer")
    ax.scatter(hs, Czs_x, color='steelblue', s=80, label='From X', zorder=3)
    ax.scatter(hs, Czs_y, color='tomato',    s=80, label='From Y', zorder=3)
    ax.axhline(Cz_final, color='black', lw=1, ls='--', label=f'Mean={Cz_final:.1f}mm')
    ax.set_xlabel("Height h (mm)"); ax.set_ylabel("Estimated Cz (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Per-layer Cx/Cy estimates
    ax = axes[1, 2]
    ax.set_title("Cx / Cy estimate per layer")
    ax.scatter(hs, Cxs, color='steelblue', s=80, label='Cx (from X)', zorder=3)
    ax.scatter(hs, Cys, color='tomato',    s=80, label='Cy (from Y)', zorder=3)
    ax.axhline(Cx_fit, color='steelblue', lw=1, ls='--', label=f'Cx fit={Cx_fit:.1f}mm')
    ax.axhline(Cy_fit, color='tomato',    lw=1, ls='--', label=f'Cy fit={Cy_fit:.1f}mm')
    ax.set_xlabel("Height h (mm)"); ax.set_ylabel("Estimated nadir position (mm)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.file).with_name("jenga_linear_fit.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
