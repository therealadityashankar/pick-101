"""Test projection accuracy during live policy rollouts.

Runs a trained PPO policy on LiftCubeProjectionEnv, and at each step:
  - Renders the wrist cam
  - Gets segmentation-based cube corners (ground truth)
  - Gets projection-based cube corners (from env._get_camera_matrices)
  - Compares them and records per-step error

Saves a side-by-side video: wrist | seg corners (green) | proj corners (blue) | both overlaid.
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

from src.envs.lift_cube_image_simplified import LiftCubeImageEnv
from src.envs.lift_cube_projection import LiftCubeProjectionEnv


# ------------------------------------------------------------------
# Camera mirroring helpers
# ------------------------------------------------------------------

def mirror_camera_noise(proj_env: LiftCubeProjectionEnv, seg_env: LiftCubeImageEnv):
    """Per-step: set seg_env model camera to match proj_env's noisy world-frame pose.

    proj_env applies noise in world frame each step (cam_xmat @ delta_R, cam_xpos + offset).
    seg_env renders using model.cam_pos/quat (body-relative). We back-calculate body-frame
    values from proj_env's noisy world-frame camera so both envs use identical viewpoints.
    """
    cam_id = proj_env._wrist_cam_id
    parent_body_id = proj_env.model.cam_bodyid[cam_id]

    # Noisy world-frame camera pose from proj_env
    cam_xmat = proj_env.data.cam_xmat[cam_id].reshape(3, 3)
    delta_R = proj_env._euler_to_mat(proj_env._cam_euler_offset)
    noisy_cam_xmat = cam_xmat @ delta_R  # (3,3) col-major: col i = world dir of cam axis i
    noisy_cam_pos = proj_env.data.cam_xpos[cam_id] + proj_env._cam_pos_offset

    # Parent body world pose
    body_xpos = proj_env.data.xpos[parent_body_id]    # (3,)
    body_xmat = proj_env.data.xmat[parent_body_id].reshape(3, 3)  # world frame

    # Back-calculate body-relative camera pos and rot
    body_cam_pos = body_xmat.T @ (noisy_cam_pos - body_xpos)
    body_cam_rot = body_xmat.T @ noisy_cam_xmat  # body-frame rotation matrix

    # Convert body-frame rotation to quaternion (MuJoCo uses w,x,y,z)
    seg_env.model.cam_pos[seg_env._wrist_cam_id] = body_cam_pos
    seg_env.model.cam_quat[seg_env._wrist_cam_id] = _rot_to_quat(body_cam_rot)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to MuJoCo quaternion (w,x,y,z)."""
    # Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# ------------------------------------------------------------------
# Segmentation helpers (from LiftCubeImageEnv)
# ------------------------------------------------------------------

def get_seg_corners_px(seg_env: LiftCubeImageEnv, crop_x: int):
    """Return (4, 2) corners in cropped pixel space from segmentation, or None."""
    id_map = seg_env._render_seg(hide_cube=False)
    ys, xs = np.where(id_map == seg_env._cube_top_face_geom_id)
    if len(ys) < 4:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
    return approx.astype(np.float32)


def draw_corners_px(img, corners_px, color, label):
    out = img.copy()
    pts = corners_px.astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    for x, y in pts:
        cv2.circle(out, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    config_path = run_dir / "config.yaml"
    env_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            env_cfg = yaml.safe_load(f).get("env", {})

    # Find latest checkpoint
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("*.zip"),
        key=lambda p: int(next((x for x in p.stem.split("_") if x.isdigit()), 0))
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints in {run_dir / 'checkpoints'}")
    model_path = checkpoints[-1]
    print(f"Using checkpoint: {model_path}")

    normalize_path = run_dir / "vec_normalize.pkl"

    # Projection env — policy runs here
    proj_env = LiftCubeProjectionEnv(
        curriculum_stage=env_cfg.get("curriculum_stage", 1),
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v11"),
        lock_wrist=env_cfg.get("lock_wrist", True),
        cam_pos_noise=env_cfg.get("cam_pos_noise", 0.003),
        cam_euler_noise=env_cfg.get("cam_euler_noise", 0.05),
        obs_noise=0.0,  # zero obs noise so comparison is clean
    )
    vec_env = DummyVecEnv([lambda: proj_env])
    if normalize_path.exists():
        vec_env = VecNormalize.load(str(normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Segmentation env — mirrors proj_env state for ground-truth corners
    seg_env = LiftCubeImageEnv(
        render_mode="rgb_array",
        curriculum_stage=env_cfg.get("curriculum_stage", 1),
        cam_pos_noise=0.0,
        cam_euler_noise=0.0,
        obs_noise=0.0,
    )

    model = PPO.load(str(model_path))
    print(f"Loaded model from {model_path}")

    # Use seg_env's model for rendering — it has the noisy camera applied per-step,
    # so the display frame matches the coordinate system used by both seg and proj corners.
    wrist_renderer = mujoco.Renderer(seg_env.model, height=480, width=640)
    crop_x = proj_env._crop_x

    all_errors = []
    frames = []
    output_path = Path(args.output) if args.output else run_dir / "proj_policy_test.mp4"

    for ep in range(args.episodes):
        seed = ep
        obs = vec_env.reset()
        # Mirror proj_env state into seg_env so segmentation sees the same pose
        seg_env.reset(seed=seed)

        ep_errors = []

        for step in range(env_cfg.get("max_episode_steps", 200)):
            # Mirror proj_env physics state into seg_env
            seg_env.data.qpos[:] = proj_env.data.qpos[:]
            seg_env.data.qvel[:] = proj_env.data.qvel[:]
            mujoco.mj_forward(seg_env.model, seg_env.data)
            # Mirror camera noise per-step (proj_env applies noise in world frame each step)
            mirror_camera_noise(proj_env, seg_env)
            mujoco.mj_forward(seg_env.model, seg_env.data)

            # --- Segmentation corners (ground truth, zero noise) ---
            seg_corners = get_seg_corners_px(seg_env, crop_x)

            # --- Projection corners from env (same code as training) ---
            K, R, t = proj_env._get_camera_matrices()
            world_corners = proj_env._get_cube_top_corners_world()
            proj_norm = proj_env._project_points_norm(world_corners, K, R, t)
            proj_px = proj_norm * np.array([proj_env.CAM_H, proj_env.CAM_H])

            # --- Per-step error ---
            if seg_corners is not None and len(seg_corners) == 4:
                errors = []
                for sp in seg_corners:
                    dists = np.linalg.norm(proj_px - sp, axis=1)
                    errors.append(float(np.min(dists)))
                mean_err = np.mean(errors)
                ep_errors.append(mean_err)
                all_errors.append(mean_err)

            # --- Render wrist frame ---
            wrist_renderer.update_scene(seg_env.data, camera="wrist_cam")
            frame = wrist_renderer.render()[:, crop_x:crop_x + 480, :]

            raw = frame.copy()
            seg_img = draw_corners_px(frame, seg_corners, (0, 220, 0), "Seg") if seg_corners is not None else frame.copy()
            proj_img = draw_corners_px(frame, proj_px, (0, 100, 255), "Proj")
            both = draw_corners_px(seg_img, proj_px, (0, 100, 255), "Both") if seg_corners is not None else proj_img.copy()

            # Error annotation
            if ep_errors:
                cv2.putText(both, f"err={ep_errors[-1]:.1f}px", (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            frames.append(np.concatenate([raw, seg_img, proj_img, both], axis=1))

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            if done[0]:
                break

        print(f"Episode {ep+1}: {len(ep_errors)} steps, "
              f"mean err={np.mean(ep_errors):.2f}px, max={np.max(ep_errors):.2f}px")

    if frames:
        imageio.mimsave(str(output_path), frames, fps=30)
        print(f"\nSaved video to {output_path}")

    print(f"\n{'='*55}")
    print(f"Summary across {len(all_errors)} steps:")
    print(f"  Mean error : {np.mean(all_errors):.2f} px")
    print(f"  Std  error : {np.std(all_errors):.2f} px")
    print(f"  Min  error : {np.min(all_errors):.2f} px")
    print(f"  Max  error : {np.max(all_errors):.2f} px")
    print(f"{'='*55}")

    wrist_renderer.close()
    proj_env.close()
    seg_env.close()


if __name__ == "__main__":
    main()
