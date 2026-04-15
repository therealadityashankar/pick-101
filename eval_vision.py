"""Evaluate a trained vision-based lift policy with video recording."""
import argparse
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube_image_simplified import LiftCubeImageEnv


def _seg_obs_from_renderer(seg_renderer, model, data, geom_ids: dict, hide_geom_ids: list, crop_x: int) -> dict:
    """Render segmentation, hiding specified geoms so fingertips aren't occluded by the cube."""
    # Temporarily move cube geoms to hidden group 4
    orig_groups = [model.geom_group[gid] for gid in hide_geom_ids]
    for gid in hide_geom_ids:
        model.geom_group[gid] = 4

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[4] = 0

    seg_renderer.update_scene(data, camera="wrist_cam", scene_option=scene_option)
    seg_renderer.enable_segmentation_rendering()
    seg = seg_renderer.render()
    seg_renderer.disable_segmentation_rendering()

    # Restore groups
    for gid, orig in zip(hide_geom_ids, orig_groups):
        model.geom_group[gid] = orig

    geom_id_map = seg[..., 0]
    w_full = geom_id_map.shape[1]
    cropped = geom_id_map[:, crop_x: w_full - crop_x]
    return {name: (cropped == gid) for name, gid in geom_ids.items()}


def _topmost_point(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    idx = np.argmin(ys)
    return int(xs[idx]), int(ys[idx])


def _cube_corners_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2).astype(int)


def draw_obs_from_array(frame, raw_obs: np.ndarray) -> np.ndarray:
    """Draw keypoints directly from the 13-dim obs array (noisy, as seen by the model).

    raw_obs: [su, sv, mu, mv, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, gripper]
    All coords in [0, 1] normalised image space.
    """
    img = frame.copy()
    h, w = img.shape[:2]

    def to_px(u, v):
        return int(round(u * w)), int(round(v * h))

    # Static fingertip
    su, sv = raw_obs[0], raw_obs[1]
    pt = to_px(su, sv)
    cv2.circle(img, pt, 6, (0, 80, 255), -1)
    cv2.putText(img, "S", (pt[0] + 8, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

    # Moving fingertip
    mu, mv = raw_obs[2], raw_obs[3]
    pt = to_px(mu, mv)
    cv2.circle(img, pt, 6, (255, 80, 0), -1)
    cv2.putText(img, "M", (pt[0] + 8, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1, cv2.LINE_AA)

    # Cube corners
    corners = raw_obs[4:12].reshape(4, 2)
    scaled = np.array([to_px(c[0], c[1]) for c in corners])
    cv2.polylines(img, [scaled], isClosed=True, color=(0, 220, 0), thickness=2)
    for cx, cy in scaled:
        cv2.circle(img, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)

    # Top-left visibility indicators (always visible since drawn from obs)
    for i, (lbl, color) in enumerate([("S", (0, 80, 255)), ("M", (255, 80, 0)), ("C", (0, 220, 0))]):
        x = 8 + i * 28
        cv2.circle(img, (x, 12), 8, color, -1)
        cv2.putText(img, lbl, (x - 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_obs_overlay(frame, seg_masks: dict, scale_x: float, scale_y: float) -> np.ndarray:
    img = frame.copy()

    def scale(pt):
        if pt is None:
            return None
        return int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))

    corners = _cube_corners_from_mask(seg_masks.get("cube", np.zeros((1, 1), bool)))
    if corners is not None:
        scaled_corners = np.array([scale(c) for c in corners])
        cv2.polylines(img, [scaled_corners], isClosed=True, color=(0, 220, 0), thickness=2)
        for cx, cy in scaled_corners:
            cv2.circle(img, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)

    for key, color, label in [
        ("static_pad", (0, 80, 255), "S"),
        ("moving_pad", (255, 80, 0), "M"),
    ]:
        pt = scale(_topmost_point(seg_masks.get(key, np.zeros((1, 1), bool))))
        if pt is not None:
            cv2.circle(img, pt, 6, color, -1)
            cv2.putText(img, label, (pt[0] + 8, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Top-left visibility indicators
    indicators = [
        ("S", seg_masks.get("static_pad", np.zeros((1,1), bool)).sum() > 0, (0, 80, 255)),
        ("M", seg_masks.get("moving_pad", np.zeros((1,1), bool)).sum() > 0, (255, 80, 0)),
        ("C", seg_masks.get("cube", np.zeros((1,1), bool)).sum() > 0, (0, 220, 0)),
    ]
    for i, (lbl, visible, color) in enumerate(indicators):
        x = 8 + i * 28
        dot_color = color if visible else (60, 60, 60)
        cv2.circle(img, (x, 12), 8, dot_color, -1)
        cv2.putText(img, lbl, (x - 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Path to run directory (loads config, model, and normalize from there)")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--model", type=str, default=None, help="Path to model .zip file (overrides --run)")
    parser.add_argument("--normalize", type=str, default=None, help="Path to vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.run is None and args.model is None:
        parser.error("Either --run or --model must be specified")

    env_cfg = {}
    run_dir = None
    if args.run:
        run_dir = Path(args.run)
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            env_cfg = config.get("env", {})
            print(f"Loaded config from {config_path}")

        if args.model is None:
            checkpoints = list((run_dir / "checkpoints").glob("*.zip"))

            def get_step(p):
                for part in p.stem.split("_"):
                    if part.isdigit():
                        return int(part)
                return 0

            if args.checkpoint:
                model_path = next(
                    (p for p in checkpoints if str(args.checkpoint) in p.stem), None
                )
                if model_path is None:
                    raise ValueError(f"Checkpoint {args.checkpoint} not found")
            else:
                checkpoints = sorted(checkpoints, key=get_step)
                model_path = checkpoints[-1] if checkpoints else None
                if model_path is None:
                    raise ValueError(f"No checkpoints found in {run_dir / 'checkpoints'}")
            args.model = str(model_path)

        if args.normalize is None:
            normalize_path = run_dir / "vec_normalize.pkl"
            if normalize_path.exists():
                args.normalize = str(normalize_path)

    # Output path
    if args.output is None:
        model_path = Path(args.model)
        if run_dir is None:
            run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
        step_str = ""
        for part in model_path.stem.split("_"):
            if part.isdigit():
                step_str = f"_{part}"
                break
        output_path = run_dir / f"eval{step_str}.mp4"
    else:
        output_path = Path(args.output)

    # Create env
    env = LiftCubeImageEnv(
        render_mode="rgb_array",
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v11"),
        curriculum_stage=env_cfg.get("curriculum_stage", 1),
        lock_wrist=env_cfg.get("lock_wrist", True),
        cam_pos_noise=env_cfg.get("cam_pos_noise", 0.003),
        cam_euler_noise=env_cfg.get("cam_euler_noise", 0.05),
        obs_noise=env_cfg.get("obs_noise", 0.01),
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization from {args.normalize}")

    model = SAC.load(args.model)
    print(f"Loaded model from {args.model}")

    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer = mujoco.Renderer(env.model, height=480, width=640)
    _crop_x = (640 - 480) // 2

    _seg_geom_ids = {
        "static_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "static_fingertip_marker"),
        "moving_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_fingertip_marker"),
        "cube":       mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face"),
    }
    _hide_geom_ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face"),
    ]
    print(f"Geom IDs: {_seg_geom_ids}")

    frames_closeup = []
    frames_wrist = []
    frames_obs_viz = []
    total_rewards = []
    successes = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0

        for step in range(env_cfg.get("max_episode_steps", 200)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

            i = info[0]
            act = action[0]
            print(f"  step={step:3d}: z={i.get('cube_z', 0):.4f}, "
                  f"r={reward[0]:+.3f}, "
                  f"act=[{act[0]:+.2f},{act[1]:+.2f},{act[2]:+.2f},{act[3]:+.2f}], "
                  f"grasp={i.get('is_grasping', False)}")

            # Closeup render
            frame_closeup = env.render(camera="closeup")
            if frame_closeup is not None:
                frames_closeup.append(frame_closeup)

            # Wrist cam render
            wrist_renderer.update_scene(env.data, camera="wrist_cam")
            wrist_frame = wrist_renderer.render()
            crop_x = (640 - 480) // 2
            wrist_frame = wrist_frame[:, crop_x:crop_x + 480, :]
            if frame_closeup is not None:
                h, w = frame_closeup.shape[:2]
                wrist_frame = cv2.resize(wrist_frame, (w, h), interpolation=cv2.INTER_AREA)
            frames_wrist.append(wrist_frame)

            # Obs viz: draw directly from raw (noisy) obs as seen by the model
            raw_obs = env._get_obs()
            out_h, out_w = wrist_frame.shape[:2]
            annotated = draw_obs_from_array(wrist_frame, raw_obs)
            frames_obs_viz.append(np.concatenate([wrist_frame, annotated], axis=1))

            if done[0]:
                break

        final_info = info[0]
        is_success = final_info.get("is_success", False)
        successes.append(is_success)
        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, success={is_success}, "
              f"cube_z={final_info.get('cube_z', 0):.3f}")

    if frames_closeup:
        closeup_path = output_path.with_stem(output_path.stem + "_closeup")
        imageio.mimsave(str(closeup_path), frames_closeup, fps=args.fps)
        print(f"\nSaved closeup video to {closeup_path}")
    if frames_wrist:
        wrist_path = output_path.with_stem(output_path.stem + "_wrist")
        imageio.mimsave(str(wrist_path), frames_wrist, fps=args.fps)
        print(f"Saved wrist video to {wrist_path}")
    if frames_obs_viz:
        obs_viz_path = output_path.with_stem(output_path.stem + "_obs_viz")
        imageio.mimsave(str(obs_viz_path), frames_obs_viz, fps=args.fps)
        print(f"Saved obs viz video to {obs_viz_path}")

    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate: {100 * np.mean(successes):.1f}%")

    wrist_renderer.close()
    seg_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
