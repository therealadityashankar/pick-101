"""plot_calibration_efficiency.py — How many calibration points / layers do you need?

Tests all combinations of 2, 3, 4 height layers × K positions per layer.

Usage:
    uv run python plot_calibration_efficiency.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

N_TRIALS        = 100
MAX_OUTLIER_ERR = 50.0
RNG             = np.random.default_rng(42)


def fit_from_subset(true_x, true_y, obs_x, obs_y, heights):
    """Per-layer linear fit → mean Cx, Cy, Cz."""
    layers = [h for h in sorted(set(heights.tolist())) if h > 0]
    if len(layers) < 2:
        return None
    Cxs, Cys, Czs = [], [], []
    for h in layers:
        m = heights == h
        if m.sum() < 2:
            continue
        ex = obs_x[m] - true_x[m]
        ey = obs_y[m] - true_y[m]
        px = np.polyfit(true_x[m], ex, 1)
        py = np.polyfit(true_y[m], ey, 1)
        sx, bx = px[0], px[1]
        sy, by = py[0], py[1]
        if sx <= 0 or sy <= 0:
            continue
        Cxs.append(-bx / sx)
        Cys.append(-by / sy)
        Czs.append(h * (1 + 1/sx))
        Czs.append(h * (1 + 1/sy))
    if len(Czs) < 2:
        return None
    return np.mean(Cxs), np.mean(Cys), np.mean(Czs)


def eval_errors(true_x, true_y, obs_x, obs_y, heights, Cx, Cy, Cz):
    mask  = heights > 0
    scale = Cz / (Cz - heights[mask])
    cx    = Cx + (obs_x[mask] - Cx) / scale
    cy    = Cy + (obs_y[mask] - Cy) / scale
    err   = np.sqrt((cx - true_x[mask])**2 + (cy - true_y[mask])**2)
    return err.mean(), err.std()


def run_experiment(tx, ty, ox, oy, hs, true_x, true_y, obs_x, obs_y, heights,
                   layer_subset, max_k):
    layer_indices = {h: np.where(hs == h)[0] for h in layer_subset}
    K_values = list(range(2, max_k + 1))
    results  = []
    for k in K_values:
        if any(len(v) < k for v in layer_indices.values()):
            break
        trial_errs, trial_params = [], []
        for _ in range(N_TRIALS):
            idx = np.concatenate([
                RNG.choice(layer_indices[h], size=k, replace=False)
                for h in layer_subset
            ])
            result = fit_from_subset(tx[idx], ty[idx], ox[idx], oy[idx], hs[idx])
            if result is None:
                continue
            Cx_f, Cy_f, Cz_f = result
            m, _ = eval_errors(true_x, true_y, obs_x, obs_y, heights, Cx_f, Cy_f, Cz_f)
            if m > MAX_OUTLIER_ERR:
                continue
            trial_errs.append(m)
            trial_params.append((Cx_f, Cy_f, Cz_f))
        if len(trial_errs) < 5:
            continue
        arr = np.array(trial_errs)
        par = np.array(trial_params)
        results.append({
            'k': k, 'n': k * len(layer_subset),
            'mean':   np.mean(arr),
            'median': np.median(arr),
            'p25':    np.percentile(arr, 25),
            'p75':    np.percentile(arr, 75),
            'dCz':    np.std(par[:, 2]),
        })
    return results


def main():
    d       = np.load("calibration_jenga.npz")
    true_x  = d['true_x'].astype(float)
    true_y  = d['true_y'].astype(float)
    obs_x   = d['obs_x'].astype(float)
    obs_y   = d['obs_y'].astype(float)
    heights = d['heights'].astype(float)

    nonzero     = heights > 0
    tx, ty      = true_x[nonzero], true_y[nonzero]
    ox, oy      = obs_x[nonzero],  obs_y[nonzero]
    hs          = heights[nonzero]
    all_layers  = sorted(set(hs.tolist()))   # [15, 30, 45, 60]
    pts_per_lyr = min((hs == h).sum() for h in all_layers)

    base_result = fit_from_subset(tx, ty, ox, oy, hs)
    Cx_b, Cy_b, Cz_b = base_result
    base_mean, _ = eval_errors(true_x, true_y, obs_x, obs_y, heights, Cx_b, Cy_b, Cz_b)
    print(f"Baseline: Cx={Cx_b:.1f} Cy={Cy_b:.1f} Cz={Cz_b:.1f}  error={base_mean:.3f}mm\n")

    # All combos of 2, 3, 4 layers
    layer_configs = []
    for n in [2, 3, 4]:
        for combo in combinations(all_layers, n):
            layer_configs.append(list(combo))

    n_layer_colors = {2: '#e05050', 3: '#e0a030', 4: '#4a90d9'}

    print(f"{'Layers':>22}  {'K':>3}  {'N':>4}  {'mean':>6}  {'p25':>6}  {'p75':>6}")
    all_results = {}
    for combo in layer_configs:
        label = "h=" + "+".join(str(int(h)) for h in combo)
        res   = run_experiment(tx, ty, ox, oy, hs, true_x, true_y, obs_x, obs_y,
                               heights, combo, int(pts_per_lyr))
        all_results[label] = (combo, res)
        for r in res:
            print(f"{label:>22}  {r['k']:3d}  {r['n']:4d}  "
                  f"{r['mean']:6.3f}  {r['p25']:6.3f}  {r['p75']:6.3f}")
        print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Calibration efficiency — varying layers & positions\n"
        f"({N_TRIALS} random subsets per config, evaluated on all 60 non-zero points)",
        fontsize=11)

    for label, (combo, res) in all_results.items():
        if not res:
            continue
        ns    = [r['n']    for r in res]
        means = [r['mean'] for r in res]
        p25   = [r['p25']  for r in res]
        p75   = [r['p75']  for r in res]
        dCzs  = [r['dCz']  for r in res]
        nl    = len(combo)
        color = n_layer_colors[nl]
        lw    = 2.5 if nl == 4 else (1.8 if nl == 3 else 1.2)
        alpha = 0.9 if nl == 4 else (0.65 if nl == 3 else 0.45)

        axes[0].plot(ns, means, 'o-', color=color, lw=lw, ms=4, alpha=alpha)
        axes[0].fill_between(ns, p25, p75, color=color, alpha=0.04)
        axes[1].plot(ns, dCzs,  'o-', color=color, lw=lw, ms=4, alpha=alpha)

    axes[0].axhline(base_mean, color='black', lw=1.5, ls='--',
                    label=f'Baseline ({base_mean:.2f}mm)')
    axes[0].axhline(base_mean + 0.5, color='red', lw=1, ls=':',
                    label='+0.5mm above baseline')
    axes[0].axhspan(0, base_mean + 0.5, alpha=0.07, color='green')

    legend_els = [
        Line2D([0],[0], color=n_layer_colors[n], lw=2.5, label=f'{n} height layers')
        for n in [2, 3, 4]
    ] + [
        Line2D([0],[0], color='black', lw=1.5, ls='--', label=f'Baseline ({base_mean:.2f}mm)'),
        Line2D([0],[0], color='red',   lw=1,   ls=':',  label='+0.5mm above baseline'),
    ]

    axes[0].set_xlabel("Total calibration points")
    axes[0].set_ylabel("Mean |correction error| (mm)")
    axes[0].set_title("Accuracy vs total points")
    axes[0].legend(handles=legend_els, fontsize=8)
    axes[0].grid(True, alpha=0.3); axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel("Total calibration points")
    axes[1].set_ylabel("Cz uncertainty — std across random subsets (mm)")
    axes[1].set_title("Parameter stability vs total points")
    axes[1].legend(handles=legend_els[:3], fontsize=8)
    axes[1].grid(True, alpha=0.3); axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out = Path("calibration_efficiency.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
