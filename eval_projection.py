"""Evaluate a trained projection-based lift policy with video recording.

Optionally compares projection keypoints against segmentation ground truth
(--test-projection flag). When enabled, each frame shows:
  left : wrist frame + projection obs (green corners, coloured fingertips)
  right: same frame + seg corners (yellow) overlaid on projection corners (green)
Error stats are printed at the end.
"""
import argparse
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to MuJoCo quaternion (w,x,y,z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s; x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    return np.array([w, x, y, z], dtype=np.float64)


def _apply_noisy_camera(env: LiftCubeProjectionEnv):
    """Temporarily bake env's camera noise offsets into model for rendering.
    Calls mj_forward so data.cam_xpos/xmat (used by renderer) reflect the noisy pose.
    Returns (saved_pos, saved_quat) so the caller can restore afterwards.
    """
    cam_id = env._wrist_cam_id
    saved_pos = env.model.cam_pos[cam_id].copy()
    saved_quat = env.model.cam_quat[cam_id].copy()
    parent_body_id = env.model.cam_bodyid[cam_id]
    body_xpos = env.data.xpos[parent_body_id]
    body_xmat = env.data.xmat[parent_body_id].reshape(3, 3)
    noisy_world_pos = env.data.cam_xpos[cam_id] + env._cam_pos_offset
    noisy_cam_xmat = env.data.cam_xmat[cam_id].reshape(3, 3) @ env._euler_to_mat(env._cam_euler_offset)
    env.model.cam_pos[cam_id] = body_xmat.T @ (noisy_world_pos - body_xpos)
    env.model.cam_quat[cam_id] = _rot_to_quat(body_xmat.T @ noisy_cam_xmat)
    mujoco.mj_forward(env.model, env.data)
    return saved_pos, saved_quat


def _restore_camera(env: LiftCubeProjectionEnv, saved_pos, saved_quat):
    cam_id = env._wrist_cam_id
    env.model.cam_pos[cam_id] = saved_pos
    env.model.cam_quat[cam_id] = saved_quat
    mujoco.mj_forward(env.model, env.data)


def _mirror_camera_to_seg(env: LiftCubeProjectionEnv, seg_env):
    """Mirror env's noisy camera into seg_env's model (for segmentation rendering)."""
    cam_id = env._wrist_cam_id
    parent_body_id = env.model.cam_bodyid[cam_id]
    body_xpos = env.data.xpos[parent_body_id]
    body_xmat = env.data.xmat[parent_body_id].reshape(3, 3)
    noisy_world_pos = env.data.cam_xpos[cam_id] + env._cam_pos_offset
    noisy_cam_xmat = env.data.cam_xmat[cam_id].reshape(3, 3) @ env._euler_to_mat(env._cam_euler_offset)
    seg_env.model.cam_pos[seg_env._wrist_cam_id] = body_xmat.T @ (noisy_world_pos - body_xpos)
    seg_env.model.cam_quat[seg_env._wrist_cam_id] = _rot_to_quat(body_xmat.T @ noisy_cam_xmat)


def get_seg_corners_px(seg_env, crop_x: int):
    """Return (N, 2) corners in cropped pixel space from segmentation, or None."""
    id_map = seg_env._render_seg(hide_cube=False)
    ys, xs = np.where(id_map == seg_env._cube_top_face_geom_id)
    if len(ys) < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
    return approx.astype(np.float32)


def draw_obs_from_array(frame, raw_obs: np.ndarray, seg_corners_norm=None, proj_err=None,
                        extra_labels: list[tuple[str, str]] | None = None) -> np.ndarray:
    """Draw keypoints from the obs array.  Optionally overlay seg corners and error.

    raw_obs: [su, sv, mu, mv, c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y, gripper, ...]
    seg_corners_norm: (N,2) normalised [0,1] coords of seg ground-truth corners, or None
    proj_err: float mean pixel error to annotate, or None
    extra_labels: list of (name, value_str) pairs to show in top-right corner
    All coords are normalised so this works at any frame resolution.
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

    # Projection cube corners (green) + cube center (white)
    corners = raw_obs[4:12].reshape(4, 2)
    scaled = np.array([to_px(c[0], c[1]) for c in corners])
    cv2.polylines(img, [scaled], isClosed=True, color=(0, 220, 0), thickness=2)
    for cx, cy in scaled:
        cv2.circle(img, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)
    cube_center_norm = corners.mean(axis=0)
    cube_center_px = to_px(cube_center_norm[0], cube_center_norm[1])
    cv2.circle(img, cube_center_px, 6, (255, 255, 255), -1)
    cv2.putText(img, "CC", (cube_center_px[0] + 6, cube_center_px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Gripper midpoint (cyan)
    grip_mid_norm = np.array([(su + mu) / 2.0, (sv + mv) / 2.0])
    grip_mid_px = to_px(grip_mid_norm[0], grip_mid_norm[1])
    cv2.circle(img, grip_mid_px, 6, (0, 220, 220), -1)
    cv2.putText(img, "GM", (grip_mid_px[0] + 6, grip_mid_px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1, cv2.LINE_AA)

    # Line between cube center and gripper midpoint
    cv2.line(img, cube_center_px, grip_mid_px, (200, 200, 200), 1, cv2.LINE_AA)

    # Compute center_dist and show on top-left
    center_dist = float(np.linalg.norm(grip_mid_norm - cube_center_norm))
    threshold = 0.10
    pct_of_threshold = center_dist / threshold * 100.0
    ok = center_dist < threshold
    dist_color = (0, 220, 0) if ok else (0, 80, 220)
    cv2.putText(img, f"zone_err={center_dist*100:.1f}% ({pct_of_threshold:.0f}% of thresh)",
                (8, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dist_color, 1, cv2.LINE_AA)
    cv2.putText(img, "ZONED" if ok else "NOT ZONED",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dist_color, 1, cv2.LINE_AA)

    # Seg corners (yellow), if provided — normalised coords so resolution-independent
    if seg_corners_norm is not None:
        pts = np.array([to_px(c[0], c[1]) for c in seg_corners_norm])
        cv2.polylines(img, [pts], isClosed=True, color=(255, 220, 0), thickness=2)
        for cx, cy in pts:
            cv2.circle(img, (cx, cy), 4, (255, 220, 0), -1, cv2.LINE_AA)

    # Error annotation
    if proj_err is not None:
        cv2.putText(img, f"proj_err={proj_err:.1f}px", (8, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # Top-left legend dots
    for i, (lbl, color) in enumerate([("S", (0, 80, 255)), ("M", (255, 80, 0)), ("C", (0, 220, 0)),
                                       ("CC", (255, 255, 255)), ("GM", (0, 220, 220))]):
        x = 8 + i * 28
        cv2.circle(img, (x, 12), 8, color, -1)
        cv2.putText(img, lbl[0], (x - 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)

    # Extra labels (top-right corner)
    if extra_labels:
        for i, (name, val) in enumerate(extra_labels):
            text = f"{name}={val}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(img, text, (w - tw - 6, 14 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 60), 1, cv2.LINE_AA)

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
    parser.add_argument("--test-projection", action="store_true",
                        help="Mirror state into a seg env each step and overlay seg corners for error measurement")
    args = parser.parse_args()

    if args.run is None and args.model is None:
        parser.error("Either --run or --model must be specified")

    env_cfg = {}
    run_dir = None
    if args.run:
        run_dir = Path(args.run)
        # If the path doesn't exist, treat it as an experiment name and find the latest timestamp
        if not run_dir.exists():
            candidates = sorted((Path("runs") / run_dir).glob("*"), key=lambda p: p.name)
            candidates = [p for p in candidates if p.is_dir()]
            if not candidates:
                raise ValueError(f"No runs found under runs/{args.run}")
            run_dir = candidates[-1]
            print(f"Auto-selected latest run: {run_dir}")
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
                    final = run_dir / "final_model.zip"
                    if final.exists():
                        model_path = final
                    else:
                        raise ValueError(f"No checkpoints or final_model.zip found in {run_dir}")
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

    # Create projection env (select class based on config)
    if env_cfg.get("env_class") == "t1v2":
        EnvClass = LiftCubeProjectionT1V2Env
        extra_kwargs = {
            "cube_size_range": tuple(env_cfg.get("cube_size_range", [0.011, 0.019])),
            "cube_height_range": tuple(env_cfg.get("cube_height_range", [0.010, 0.035])),
            "fov_noise": env_cfg.get("fov_noise", 7.0),
        }
    else:
        EnvClass = LiftCubeProjectionEnv
        extra_kwargs = {}

    env = EnvClass(
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
        straddle_states_path=env_cfg.get("straddle_states_path", None),
        freeze_cube=env_cfg.get("freeze_cube", False),
        **extra_kwargs,
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization from {args.normalize}")

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")

    # Optional seg env for ground-truth comparison
    seg_env = None
    if args.test_projection:
        from src.envs.lift_cube_image_simplified import LiftCubeImageEnv
        seg_env = LiftCubeImageEnv(
            render_mode="rgb_array",
            curriculum_stage=env_cfg.get("curriculum_stage", 1),
            cam_pos_noise=0.0,
            cam_euler_noise=0.0,
            obs_noise=0.0,
        )
        print("Seg env created for ground-truth comparison.")

    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    _crop_x = env._crop_x

    frames_closeup = []
    frames_wrist = []
    frames_obs_viz = []
    total_rewards = []
    successes = []
    all_proj_errors = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0
        ep_errors = []

        if seg_env is not None:
            seg_env.reset(seed=ep)

        for step in range(env_cfg.get("max_episode_steps", 200)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

            i = info[0]
            act = action[0]

            # --- Seg comparison (optional) ---
            seg_corners_norm = None
            proj_err = None
            if seg_env is not None:
                seg_env.data.qpos[:] = env.data.qpos[:]
                seg_env.data.qvel[:] = env.data.qvel[:]
                mujoco.mj_forward(seg_env.model, seg_env.data)
                _mirror_camera_to_seg(env, seg_env)
                mujoco.mj_forward(seg_env.model, seg_env.data)
                seg_corners_px = get_seg_corners_px(seg_env, _crop_x)

                if seg_corners_px is not None:
                    # Normalise to [0,1] for resolution-independent drawing
                    seg_corners_norm = seg_corners_px / env.CAM_H

                # Compare seg corners against projection corners (from raw obs, pixel space)
                if seg_corners_px is not None and len(seg_corners_px) == 4:
                    raw_obs_cmp = env._get_obs()
                    proj_corners_norm = raw_obs_cmp[4:12].reshape(4, 2)  # normalised [0,1]
                    proj_px = proj_corners_norm * env.CAM_H  # pixel space
                    errors = [float(np.min(np.linalg.norm(proj_px - sp, axis=1)))
                              for sp in seg_corners_px]
                    proj_err = float(np.mean(errors))
                    ep_errors.append(proj_err)
                    all_proj_errors.append(proj_err)

            print(f"  step={step:3d}: z={i.get('cube_z', 0):.4f}, "
                  f"r={reward[0]:+.3f}, "
                  f"act=[{act[0]:+.2f},{act[1]:+.2f},{act[2]:+.2f},{act[3]:+.2f}], "
                  f"grasp={i.get('is_grasping', False)}"
                  + (f", proj_err={proj_err:.1f}px" if proj_err is not None else ""))

            # Closeup render
            frame_closeup = env.render(camera="closeup")
            if frame_closeup is not None:
                frames_closeup.append(frame_closeup)

            # Wrist cam render — bake camera noise into model temporarily so frame
            # matches the noisy projection coordinate system
            saved_pos, saved_quat = _apply_noisy_camera(env)
            wrist_renderer.update_scene(env.data, camera="wrist_cam")
            wrist_frame_raw = wrist_renderer.render()[:, _crop_x:_crop_x + 480, :]
            _restore_camera(env, saved_pos, saved_quat)

            # wrist video: resize to match closeup for side-by-side if needed
            wrist_frame_display = wrist_frame_raw
            if frame_closeup is not None:
                h_c, w_c = frame_closeup.shape[:2]
                wrist_frame_display = cv2.resize(wrist_frame_raw, (w_c, h_c), interpolation=cv2.INTER_AREA)
            frames_wrist.append(wrist_frame_display)

            # Obs viz always at native 480x480 so normalised coords map correctly
            raw_obs = env._get_obs()
            extra_labels = None
            if len(raw_obs) >= 15:
                extra_labels = [
                    ("spread", f"{raw_obs[13]:.3f}"),
                    ("min_grip", f"{raw_obs[14]:.3f}"),
                ]
                if len(raw_obs) >= 16:
                    stage_map = {0.0: "navigate", 0.5: "grip", 1.0: "lift"}
                    stage_str = stage_map.get(round(raw_obs[15] * 2) / 2, f"{raw_obs[15]:.1f}")
                    extra_labels.append(("stage", stage_str))
                cube_half = i.get("cube_half_size")
                if cube_half is not None:
                    extra_labels.append(("cube", f"{cube_half*2*1000:.0f}mm"))
            annotated = draw_obs_from_array(wrist_frame_raw, raw_obs,
                                            seg_corners_norm=seg_corners_norm,
                                            proj_err=proj_err,
                                            extra_labels=extra_labels)
            frames_obs_viz.append(np.concatenate([wrist_frame_raw, annotated], axis=1))

            if done[0]:
                break

        final_info = info[0]
        is_success = final_info.get("is_success", False)
        successes.append(is_success)
        total_rewards.append(ep_reward)
        ep_err_str = (f", proj_err mean={np.mean(ep_errors):.2f}px max={np.max(ep_errors):.2f}px"
                      if ep_errors else "")
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, success={is_success}, "
              f"cube_z={final_info.get('cube_z', 0):.3f}{ep_err_str}")

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
    print(f"  Mean reward  : {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate : {100 * np.mean(successes):.1f}%")
    if all_proj_errors:
        print(f"  Proj error   : mean={np.mean(all_proj_errors):.2f}px  "
              f"std={np.std(all_proj_errors):.2f}px  "
              f"max={np.max(all_proj_errors):.2f}px  "
              f"({len(all_proj_errors)} steps)")

    wrist_renderer.close()
    env.close()
    if seg_env is not None:
        seg_env.close()


if __name__ == "__main__":
    main()
