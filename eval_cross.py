"""Cross-evaluation: test model across a grid of cube sizes × FOV values.

Cells are evaluated in parallel (one process per cell by default).

Outputs (inside <run_dir>/cross_eval/):
  - cube{W}mm_fov{F}deg.mp4   one video per cell
  - heatmap.png
  - results.csv

Usage:
    python eval_cross.py --run lift_proj_t1v2_s3
    python eval_cross.py --run lift_proj_t1v2_s3 --episodes 2 --cube-widths 25 30 35 --fovs 70 77 85
    python eval_cross.py --run lift_proj_t1v2_s3 --workers 4
"""
import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env
from eval_projection import draw_obs_from_array

CUBE_WIDTHS_MM = [20, 25, 30, 35, 40, 50, 60, 70]
FOV_VALUES     = [60, 65, 70, 77, 85, 90, 95]

VIDEO_H = 480
VIDEO_W = 480


def resolve_run(run_arg: str) -> Path:
    run_dir = Path(run_arg)
    if not run_dir.exists():
        candidates = sorted((Path("runs") / run_dir).glob("*"), key=lambda p: p.name)
        candidates = [p for p in candidates if p.is_dir()]
        if not candidates:
            raise ValueError(f"No runs found under runs/{run_arg}")
        run_dir = candidates[-1]
    return run_dir


def load_config(run_dir: Path, model_arg=None, normalize_arg=None):
    env_cfg = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            env_cfg = yaml.safe_load(f).get("env", {})

    if model_arg is None:
        best = run_dir / "best_model" / "best_model.zip"
        if best.exists():
            model_path = best
        else:
            checkpoints = sorted(
                (run_dir / "checkpoints").glob("*.zip"),
                key=lambda p: int(next((x for x in p.stem.split("_") if x.isdigit()), 0))
            )
            if not checkpoints:
                raise ValueError(f"No model in {run_dir}")
            model_path = checkpoints[-1]
    else:
        model_path = Path(model_arg)

    norm_path = Path(normalize_arg) if normalize_arg else run_dir / "vec_normalize.pkl"
    return model_path, norm_path if norm_path.exists() else None, env_cfg


def draw_heatmap(grid, row_labels, col_labels, title, output_path):
    cell = 80
    pad_l, pad_t, pad_r, pad_b = 90, 50, 20, 50
    W = pad_l + len(col_labels) * cell + pad_r
    H = pad_t + len(row_labels) * cell + pad_b
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = grid[r, c]
            g = int(val * 255); b = int((1 - val) * 255)
            x0 = pad_l + c * cell; y0 = pad_t + r * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell - 2, y0 + cell - 2), (b, g, 40), -1)
            pct = f"{val*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, pct, (x0 + cell//2 - tw//2, y0 + cell//2 + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "FOV (deg) →", (pad_l, pad_t - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for c, lbl in enumerate(col_labels):
        x = pad_l + c * cell + cell // 2
        (tw, _), _ = cv2.getTextSize(str(lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, str(lbl), (x - tw//2, pad_t - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(img, "cube(mm)", (4, pad_t - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    for r, lbl in enumerate(row_labels):
        cv2.putText(img, str(lbl), (4, pad_t + r * cell + cell//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(img, title, (pad_l, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.imwrite(str(output_path), img)
    print(f"Heatmap saved to {output_path}")


def _eval_cell(args_tuple):
    """Worker function: evaluate one (cube_width, fov) cell. Returns (cw, fov, sr)."""
    (cw, fov, model_path, norm_path, env_cfg, out_dir, n_episodes, fps) = args_tuple

    cube_half = cw / 2 / 1000

    def _make():
        return LiftCubeProjectionT1V2Env(
            render_mode="rgb_array",
            max_episode_steps=env_cfg.get("max_episode_steps", 150),
            action_scale=env_cfg.get("action_scale", 0.02),
            lift_height=env_cfg.get("lift_height", 0.08),
            hold_steps=env_cfg.get("hold_steps", 10),
            reward_type=env_cfg.get("reward_type", "dense"),
            reward_version=env_cfg.get("reward_version", "v_unified_straddle"),
            curriculum_stage=env_cfg.get("curriculum_stage", 3),
            lock_wrist=env_cfg.get("lock_wrist", True),
            cam_pos_noise=0.0, cam_euler_noise=0.0, obs_noise=0.0,
            cube_size_range=(cube_half, cube_half),
            cube_height_range=(cube_half, cube_half),
            fov_noise=0.0,
        )

    vec_env = DummyVecEnv([_make])
    if norm_path:
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    inner_env = vec_env.envs[0] if hasattr(vec_env, "envs") else vec_env.venv.envs[0]

    # Override FOV after env creation
    inner_env.model.cam_fovy[inner_env._wrist_cam_id] = fov
    inner_env._nominal_fovy = fov

    wrist_renderer = mujoco.Renderer(inner_env.model, height=VIDEO_H, width=VIDEO_W)
    model = PPO.load(str(model_path))

    frames    = []
    successes = []

    for ep in range(n_episodes):
        obs = vec_env.reset()

        for _ in range(env_cfg.get("max_episode_steps", 150)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = vec_env.step(action)

            overview = inner_env.render(camera="closeup")
            if overview is None:
                overview = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
            ov = cv2.cvtColor(cv2.resize(overview, (VIDEO_W, VIDEO_H)), cv2.COLOR_RGB2BGR)

            wrist_renderer.update_scene(inner_env.data, camera="wrist_cam")
            wrist_raw = wrist_renderer.render()

            raw_obs = inner_env._get_obs()
            extra_labels = [
                ("cube", f"{cw:.0f}mm"),
                ("fov",  f"{fov:.0f}°"),
                ("ep",   f"{ep+1}/{n_episodes}"),
            ]
            if len(raw_obs) >= 15:
                extra_labels += [
                    ("spread",   f"{raw_obs[13]:.3f}"),
                    ("min_grip", f"{raw_obs[14]:.3f}"),
                ]
            wst = draw_obs_from_array(wrist_raw, raw_obs, extra_labels=extra_labels)
            wst = cv2.cvtColor(wst, cv2.COLOR_RGB2BGR)

            success_now = info[0].get("is_success", False)
            grasping    = info[0].get("is_grasping", False)
            cube_z      = info[0].get("cube_z", 0.0)
            cv2.putText(ov, f"z={cube_z:.3f}  grasp={'Y' if grasping else 'N'}"
                            f"  {'SUCCESS' if success_now else ''}",
                        (4, VIDEO_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if success_now else (255, 255, 255), 1)

            frames.append(cv2.cvtColor(np.hstack([ov, wst]), cv2.COLOR_BGR2RGB))

            if done[0]:
                successes.append(info[0].get("is_success", False))
                break
        else:
            successes.append(False)

    sr = float(np.mean(successes))

    video_path = Path(out_dir) / f"cube{cw:.0f}mm_fov{fov:.0f}deg.mp4"
    if frames:
        imageio.mimsave(str(video_path), frames, fps=fps)

    wrist_renderer.close()
    vec_env.close()

    return (cw, fov, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",         type=str, default=None)
    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--normalize",   type=str, default=None)
    parser.add_argument("--episodes",    type=int, default=3)
    parser.add_argument("--fps",         type=int, default=20)
    parser.add_argument("--cube-widths", type=float, nargs="+", default=None)
    parser.add_argument("--fovs",        type=float, nargs="+", default=None)
    parser.add_argument("--workers",     type=int, default=None,
                        help="Parallel worker processes (default: number of cells)")
    args = parser.parse_args()

    if args.run is None and args.model is None:
        parser.error("Either --run or --model must be specified")

    run_dir = resolve_run(args.run) if args.run else Path(args.model).parent
    model_path, norm_path, env_cfg = load_config(run_dir, args.model, args.normalize)

    cube_widths = args.cube_widths or CUBE_WIDTHS_MM
    fovs        = args.fovs        or FOV_VALUES
    n_cells     = len(cube_widths) * len(fovs)

    out_dir = run_dir / "cross_eval"
    out_dir.mkdir(exist_ok=True)

    n_workers = args.workers or n_cells
    print(f"Model:    {model_path}")
    print(f"Output:   {out_dir}")
    print(f"Grid:     {len(cube_widths)} cube sizes × {len(fovs)} FOVs = {n_cells} cells")
    print(f"Episodes: {args.episodes}/cell  Total: {n_cells * args.episodes}")
    print(f"Workers:  {n_workers}\n")

    work_items = [
        (cw, fov, model_path, norm_path, env_cfg, out_dir, args.episodes, args.fps)
        for cw in cube_widths
        for fov in fovs
    ]

    results = {}
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        for cw, fov, sr in pool.imap_unordered(_eval_cell, work_items):
            completed += 1
            results[(cw, fov)] = sr
            print(f"[{completed}/{n_cells}] cube={cw:.0f}mm  fov={fov:.0f}°  → {sr*100:.1f}%")

    # Build grid in original order
    grid = np.zeros((len(cube_widths), len(fovs)), dtype=np.float32)
    for i, cw in enumerate(cube_widths):
        for j, fov in enumerate(fovs):
            grid[i, j] = results.get((cw, fov), 0.0)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cube_width_mm", "fov_deg", "success_rate"])
        for i, cw in enumerate(cube_widths):
            for j, fov in enumerate(fovs):
                writer.writerow([cw, fov, f"{grid[i,j]:.3f}"])

    print(f"\nCSV saved to {csv_path}")
    draw_heatmap(
        grid,
        [f"{w:.0f}" for w in cube_widths],
        [f"{f:.0f}" for f in fovs],
        f"{run_dir.name} | {args.episodes} eps/cell",
        out_dir / "heatmap.png",
    )
    worst = np.unravel_index(grid.argmin(), grid.shape)
    print(f"Worst: cube={cube_widths[worst[0]]}mm  fov={fovs[worst[1]]}°  "
          f"success={grid[worst]*100:.1f}%")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
