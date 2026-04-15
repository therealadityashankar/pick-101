"""Evaluate a trained Cartesian lift policy with video recording."""
import argparse
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube import LiftCubeCartesianEnv

# Half-size of cube (metres) — used to project bbox corners
def _seg_obs_from_renderer(seg_renderer, data, geom_ids: dict, crop_x: int) -> dict:
    """Render a segmentation frame and extract pixel masks per geom.

    Returns a dict mapping each key in geom_ids to a boolean mask (H, W)
    in the *cropped* 480×480 space.
    """
    seg_renderer.update_scene(data, camera="wrist_cam")
    seg_renderer.enable_segmentation_rendering()
    seg = seg_renderer.render()          # (H, W, 2): channel 0 = geom type, channel 1 = geom id
    seg_renderer.disable_segmentation_rendering()

    # Channel 0 = geom ID, channel 1 = objtype (always mjOBJ_GEOM for rendered geoms)
    geom_id_map = seg[..., 0]  # (H, W) int32

    w_full = geom_id_map.shape[1]       # 640
    # Apply same center crop as the RGB wrist frame
    cropped = geom_id_map[:, crop_x: w_full - crop_x]   # (480, 480)

    masks = {}
    for name, gid in geom_ids.items():
        masks[name] = (cropped == gid)
    return masks


def _topmost_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) of the topmost (smallest y) pixel in mask, or None."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    idx = np.argmin(ys)
    return int(xs[idx]), int(ys[idx])


def _cube_corners_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return (N, 2) int array of convex hull corner points, or None."""
    ys, xs = np.where(mask)
    if len(ys) < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2).astype(int)


def draw_obs_overlay(frame, seg_masks: dict, scale_x: float, scale_y: float) -> np.ndarray:
    """Draw fingertip keypoints and cube top-face corners onto a wrist-cam frame.

    seg_masks: dict from _seg_obs_from_renderer (keys: static_pad, moving_pad, cube)
    scale_x/y: ratio of output frame size to 480×480 seg space
    """
    img = frame.copy()

    def scale(pt):
        if pt is None:
            return None
        return int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))

    # --- Fingertip topmost points ---
    for key, color, label in [
        ("static_pad", (0, 80, 255), "S"),
        ("moving_pad", (255, 80, 0), "M"),
    ]:
        pt = scale(_topmost_point(seg_masks.get(key, np.zeros((1, 1), bool))))
        if pt is not None:
            cv2.circle(img, pt, 6, color, -1)
            cv2.putText(img, label, (pt[0] + 8, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # --- Cube top-face convex corners ---
    corners = _cube_corners_from_mask(seg_masks.get("cube", np.zeros((1, 1), bool)))
    if corners is not None:
        scaled_corners = np.array([scale(c) for c in corners])
        cv2.polylines(img, [scaled_corners], isClosed=True, color=(0, 220, 0), thickness=2)
        for cx, cy in scaled_corners:
            cv2.circle(img, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)
        cv2.putText(img, "cube", (scaled_corners[0][0], max(scaled_corners[0][1] - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)

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
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path. Defaults to eval.mp4 in model's run directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--curriculum-stage", type=int, default=None,
                        help="Curriculum stage (overrides config)")
    parser.add_argument("--lock-wrist", action="store_true",
                        help="Lock wrist joints (overrides config)")
    parser.add_argument("--action-scale", type=float, default=None,
                        help="Action scale (overrides config)")
    args = parser.parse_args()

    # Validate arguments
    if args.run is None and args.model is None:
        parser.error("Either --run or --model must be specified")

    # Load config and paths from run directory if specified
    env_cfg = {}
    if args.run:
        run_dir = Path(args.run)
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            env_cfg = config.get("env", {})
            print(f"Loaded config from {config_path}")

        # Find model checkpoint
        if args.model is None:
            checkpoints = list((run_dir / "checkpoints").glob("*.zip"))
            if args.checkpoint:
                # Find specific checkpoint
                model_path = run_dir / "checkpoints" / f"sac_lift_{args.checkpoint}_steps.zip"
                if not model_path.exists():
                    raise ValueError(f"Checkpoint not found: {model_path}")
            else:
                # Find latest checkpoint
                def get_step(p):
                    for part in p.stem.split("_"):
                        if part.isdigit():
                            return int(part)
                    return 0
                checkpoints = sorted(checkpoints, key=get_step)
                model_path = checkpoints[-1] if checkpoints else None
                if model_path is None:
                    raise ValueError(f"No checkpoints found in {run_dir / 'checkpoints'}")
            args.model = str(model_path)

        # Load normalize stats
        if args.normalize is None:
            normalize_path = run_dir / "vec_normalize.pkl"
            if normalize_path.exists():
                args.normalize = str(normalize_path)
    else:
        run_dir = None

    # Determine output path
    if args.output is None:
        model_path = Path(args.model)
        # Model is typically in checkpoints/ subdirectory, so go up to run dir
        if run_dir is None:
            run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
        # Extract step count from model filename for video name
        step_str = ""
        for part in model_path.stem.split("_"):
            if part.isdigit():
                step_str = f"_{part}"
                break
        output_path = run_dir / f"eval{step_str}.mp4"
    else:
        output_path = Path(args.output)

    # Get env parameters (CLI overrides config)
    curriculum_stage = args.curriculum_stage if args.curriculum_stage is not None else env_cfg.get("curriculum_stage", 0)
    lock_wrist = args.lock_wrist or env_cfg.get("lock_wrist", False)
    action_scale = args.action_scale if args.action_scale is not None else env_cfg.get("action_scale", 0.02)
    max_episode_steps = env_cfg.get("max_episode_steps", 200)
    lift_height = env_cfg.get("lift_height", 0.08)
    hold_steps = env_cfg.get("hold_steps", 10)
    reward_version = env_cfg.get("reward_version", "v7")

    # Parse place_target if present
    place_target = env_cfg.get("place_target")
    if place_target is not None:
        place_target = tuple(place_target)

    # Create env
    env = LiftCubeCartesianEnv(
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        curriculum_stage=curriculum_stage,
        lock_wrist=lock_wrist,
        action_scale=action_scale,
        lift_height=lift_height,
        hold_steps=hold_steps,
        reward_version=reward_version,
        place_target=place_target,
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization from {args.normalize}")

    # Load model
    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")
    if place_target:
        print(f"Place target: {place_target}")

    # Wrist cam renderer: 640x480 -> center crop 480x480 -> resize to match other cams
    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer = mujoco.Renderer(env.model, height=480, width=640)

    # Geom IDs needed for segmentation overlay
    _seg_geom_ids = {
        "static_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"),
        "moving_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"),
        "cube":       mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face"),
    }
    _crop_x = (640 - 480) // 2  # 80px

    frames_closeup = []
    frames_wide = []
    frames_wide2 = []
    frames_wrist = []
    frames_obs_viz = []   # side-by-side: raw wrist | annotated wrist
    total_rewards = []
    successes = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0

        for step in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

            # Log every step with z height, action, and reward
            i = info[0]
            act = action[0]  # Get raw action
            log_line = (f"  step={step:3d}: z={i.get('cube_z', 0):.4f}, "
                        f"r={reward[0]:+.3f}, "
                        f"act=[{act[0]:+.2f},{act[1]:+.2f},{act[2]:+.2f},{act[3]:+.2f}], "
                        f"grasp={i.get('is_grasping', False)}")
            # Add place-specific info if available
            if "cube_to_target" in i:
                log_line += f", dist={i['cube_to_target']:.3f}"
            if "is_placed" in i:
                log_line += f", placed={i['is_placed']}"
            print(log_line)

            # Render all camera views
            frame_closeup = env.render(camera="closeup")
            frame_wide = env.render(camera="wide")
            frame_wide2 = env.render(camera="wide2")
            if frame_closeup is not None:
                frames_closeup.append(frame_closeup)
            if frame_wide is not None:
                frames_wide.append(frame_wide)
            if frame_wide2 is not None:
                frames_wide2.append(frame_wide2)

            # Render wrist cam: 640x480 -> center crop 480x480 -> resize to closeup size
            wrist_renderer.update_scene(env.data, camera="wrist_cam")
            wrist_frame = wrist_renderer.render()  # (480, 640, 3)
            crop_x = (640 - 480) // 2
            wrist_frame = wrist_frame[:, crop_x:crop_x + 480, :]  # (480, 480, 3)
            if frame_closeup is not None:
                h, w = frame_closeup.shape[:2]
                wrist_frame = cv2.resize(wrist_frame, (w, h), interpolation=cv2.INTER_AREA)
            frames_wrist.append(wrist_frame)

            # Obs viz: raw wrist (left) | annotated wrist (right)
            seg_masks = _seg_obs_from_renderer(seg_renderer, env.data, _seg_geom_ids, _crop_x)
            out_h, out_w = wrist_frame.shape[:2]
            annotated = draw_obs_overlay(wrist_frame, seg_masks, out_w / 480, out_h / 480)
            frames_obs_viz.append(np.concatenate([wrist_frame, annotated], axis=1))

            if done[0]:
                break

        # Get final info
        final_info = info[0]
        is_success = final_info.get("is_success", False)
        successes.append(is_success)
        total_rewards.append(ep_reward)

        ep_summary = (f"Episode {ep + 1}: reward={ep_reward:.2f}, success={is_success}, "
                      f"cube_z={final_info.get('cube_z', 0):.3f}")
        if "cube_to_target" in final_info:
            ep_summary += f", dist_to_target={final_info['cube_to_target']:.3f}"
        else:
            ep_summary += f", gripper_to_cube={final_info.get('gripper_to_cube', 0):.3f}"
        print(ep_summary)

    # Save all videos
    if frames_closeup:
        closeup_path = output_path.with_stem(output_path.stem + "_closeup")
        imageio.mimsave(str(closeup_path), frames_closeup, fps=args.fps)
        print(f"\nSaved closeup video to {closeup_path}")
    if frames_wide:
        wide_path = output_path.with_stem(output_path.stem + "_wide")
        imageio.mimsave(str(wide_path), frames_wide, fps=args.fps)
        print(f"Saved wide video to {wide_path}")
    if frames_wide2:
        wide2_path = output_path.with_stem(output_path.stem + "_wide2")
        imageio.mimsave(str(wide2_path), frames_wide2, fps=args.fps)
        print(f"Saved wide2 video to {wide2_path}")
    if frames_wrist:
        wrist_path = output_path.with_stem(output_path.stem + "_wrist")
        imageio.mimsave(str(wrist_path), frames_wrist, fps=args.fps)
        print(f"Saved wrist cam video to {wrist_path}")
    if frames_obs_viz:
        obs_viz_path = output_path.with_stem(output_path.stem + "_obs_viz")
        imageio.mimsave(str(obs_viz_path), frames_obs_viz, fps=args.fps)
        print(f"Saved obs viz video to {obs_viz_path}")

    # Save combined video (all 4 views horizontally concatenated)
    if frames_closeup and frames_wide and frames_wide2 and frames_wrist:
        frames_combined = [
            np.concatenate([c, w, w2, wr], axis=1)
            for c, w, w2, wr in zip(frames_closeup, frames_wide, frames_wide2, frames_wrist)
        ]
        combined_path = output_path.with_stem(output_path.stem + "_combined")
        imageio.mimsave(str(combined_path), frames_combined, fps=args.fps)
        print(f"Saved combined video to {combined_path}")

    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate: {100 * np.mean(successes):.1f}%")

    wrist_renderer.close()
    seg_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
