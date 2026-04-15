"""Side-by-side calibration viewer: real wrist cam (left) vs sim render (right).

Shows projected cube corners and fingertips overlaid on both views so you can
visually align the simulation to match reality.

The sim mirrors the real robot's joint positions in real time, so you can move
the arm and see if the sim tracks correctly.

Optionally runs a trained policy on the real robot (very slowly, for safety).

Usage:
    python calibration_viewer.py --port /dev/tty.usbmodem5A680089441
    python calibration_viewer.py --port /dev/tty.usbmodem5A680089441 --camera 1
    python calibration_viewer.py --port /dev/tty.usbmodem5A680089441 \\
        --model runs/lift_proj_t1_s3/<ts>/best_model/best_model.zip \\
        --normalize runs/lift_proj_t1_s3/<ts>/vec_normalize.pkl

Controls:
    Q        — quit
    P        — save snapshot
    R        — toggle policy on real robot (requires --model)
    S        — toggle policy in sim only (requires --model)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env
from src.robot.real_robot import RealRobot, JOINT_NAMES

# HSV range for bright blue cube detection
BLUE_LOWER = np.array([90, 120, 80])
BLUE_UPPER = np.array([115, 255, 255])

# Display size for each panel
PANEL_W = 480
PANEL_H = 480

# Action scale for real robot (much slower than sim's 0.02 for safety)
# Override via --action-scale at the command line
REAL_ACTION_SCALE = 0.002


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_camera_intrinsics(path: str = "calibration/camera_intrinsics.json"):
    """Load camera matrix and distortion coefficients. Returns (K, dist) or (None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    K    = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"],   dtype=np.float64)
    print(f"Loaded camera intrinsics from {p}  (RMS={data.get('rms_error_px', '?'):.3f} px)")
    return K, dist


def undistort_frame(frame: np.ndarray, K, dist) -> np.ndarray:
    if K is None:
        return frame
    return cv2.undistort(frame, K, dist)


def detect_blue_corners(frame: np.ndarray) -> np.ndarray | None:
    """Detect bright blue blob and return 4 corners as (4,2) int array, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None
    s, d = hull.sum(1), np.diff(hull, axis=1).ravel()
    return np.array([hull[np.argmin(s)], hull[np.argmin(d)],
                     hull[np.argmax(s)], hull[np.argmax(d)]], dtype=np.int32)


def draw_quad(img: np.ndarray, corners, color, label: str = ""):
    if corners is None:
        return
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    for pt in corners:
        cv2.circle(img, tuple(np.array(pt, dtype=int)), 5, color, -1)
    if label and len(corners):
        x, y = int(corners[0][0]), int(corners[0][1]) - 8
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def norm_to_px(norm_xy, w=PANEL_W, h=PANEL_H):
    return (int(norm_xy[0] * w), int(norm_xy[1] * h))


def draw_obs_overlay(img: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Draw fingertips + cube corners from obs array onto img.

    Handles both 13-dim (base) and 16-dim (t1v2) obs.
    For 16-dim, also shows gripper_spread, min_gripper_frac, stage in the corner.
    """
    out = img.copy()
    h, w = out.shape[:2]

    def to_px(u, v):
        return int(round(u * w)), int(round(v * h))

    # Static fingertip (orange-red)
    cv2.circle(out, to_px(obs[0], obs[1]), 6, (0, 80, 255), -1)
    cv2.putText(out, "S", (to_px(obs[0], obs[1])[0] + 8, to_px(obs[0], obs[1])[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

    # Moving fingertip (blue)
    cv2.circle(out, to_px(obs[2], obs[3]), 6, (255, 80, 0), -1)
    cv2.putText(out, "M", (to_px(obs[2], obs[3])[0] + 8, to_px(obs[2], obs[3])[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1, cv2.LINE_AA)

    # Cube corners (green quad)
    corners = obs[4:12].reshape(4, 2)
    scaled = np.array([to_px(c[0], c[1]) for c in corners])
    cv2.polylines(out, [scaled], isClosed=True, color=(0, 220, 0), thickness=2)
    for cx, cy in scaled:
        cv2.circle(out, (cx, cy), 4, (0, 255, 100), -1, cv2.LINE_AA)

    # t1v2 extra dims: gripper_spread, min_gripper_frac, stage
    if len(obs) >= 16:
        gripper_spread = obs[13]
        min_frac       = obs[14]
        stage_val      = obs[15]
        stage_name = {0.0: "zone", 0.33: "straddle", 0.67: "grip", 1.0: "lift"}.get(
            round(float(stage_val), 2), f"{stage_val:.2f}"
        )
        lines = [
            f"spread={gripper_spread:.2f}",
            f"min_g={min_frac:.2f}",
            f"stage={stage_name}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(out, line, (4, h - 8 - (len(lines) - 1 - i) * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

    return out


def lerobot_to_mujoco(joint_positions: dict) -> dict:
    """Convert lerobot degree positions to MuJoCo radians."""
    scale = np.pi / 180.0
    return {
        "shoulder_pan":  (joint_positions.get("shoulder_pan",  0.0) * 1.02857 + 22.423) * scale,
        "shoulder_lift": joint_positions.get("shoulder_lift", 0.0) * scale,
        "elbow_flex":    joint_positions.get("elbow_flex",    0.0) * scale,
        "wrist_flex":    joint_positions.get("wrist_flex",    0.0) * scale,
        "wrist_roll":    -joint_positions.get("wrist_roll",   0.0) * scale + 1.679,
        "gripper":       np.radians((joint_positions.get("gripper", 0.0) - 7.852) / 0.648),
    }


def set_sim_joints(env: LiftCubeProjectionEnv, joint_rad: dict):
    for name, angle in joint_rad.items():
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            addr = env.model.jnt_qposadr[jid]
            env.data.qpos[addr] = angle
    mujoco.mj_forward(env.model, env.data)


def render_sim(env: LiftCubeProjectionEnv) -> np.ndarray:
    if env._renderer is None:
        env._renderer = mujoco.Renderer(env.model, height=env.CAM_H, width=env.CAM_W)
    env._renderer.update_scene(env.data, camera="wrist_cam")
    img = env._renderer.render()
    crop_x = (env.CAM_W - env.CAM_H) // 2
    img = img[:, crop_x:crop_x + env.CAM_H]
    img = cv2.resize(img, (PANEL_W, PANEL_H))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_sim_overview(env: LiftCubeProjectionEnv) -> np.ndarray:
    if not hasattr(env, "_overview_renderer") or env._overview_renderer is None:
        env._overview_renderer = mujoco.Renderer(env.model, height=PANEL_H, width=PANEL_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.15, 0.0, 0.15]
    cam.distance = 0.9
    cam.azimuth = 150
    cam.elevation = -20
    env._overview_renderer.update_scene(env.data, camera=cam)
    img = env._overview_renderer.render()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_base_front_x(env: LiftCubeProjectionEnv) -> float:
    """Return world-space x of the forward-most point of the robot base body."""
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    max_x = 0.0
    for i in range(env.model.ngeom):
        if env.model.geom_bodyid[i] == base_id:
            gx = env.model.geom_pos[i, 0]
            hx = env.model.geom_size[i, 0]  # half-size x for box geoms
            max_x = max(max_x, gx + hx)
    return max_x


def get_gripper_qpos_deg(env: LiftCubeProjectionEnv) -> float:
    """Read actual gripper joint angle from qpos (degrees). Use this instead of ctrl[5]
    because ctrl jumps instantly to the target while qpos reflects actual position."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    return float(np.degrees(env.data.qpos[env.model.jnt_qposadr[jid]]))


def read_sim_joint_angles(env: LiftCubeProjectionEnv) -> dict:
    """Read actual sim joint angles from qpos (radians)."""
    angles = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            angles[name] = float(env.data.qpos[env.model.jnt_qposadr[jid]])
    return angles


def pin_cube(env: LiftCubeProjectionEnv, x: float, y: float):
    """Force cube to a fixed XY position each frame (overrides physics).
    Z is derived from the cube's current half-size so it sits on the floor.
    """
    cube_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    addr = env.model.jnt_qposadr[cube_joint_id]
    z = float(env._cube_half_z)  # sits flush on the floor
    env.data.qpos[addr:addr + 3] = [x, y, z]
    env.data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]  # upright quaternion
    env.data.qvel[env.model.jnt_dofadr[cube_joint_id]:env.model.jnt_dofadr[cube_joint_id] + 6] = 0
    mujoco.mj_forward(env.model, env.data)


def get_cam_info(env: LiftCubeProjectionEnv, base_front_x: float) -> dict:
    """Return sim camera pose values for display."""
    cam_id = env._wrist_cam_id
    cam_pos = env.model.cam_pos[cam_id].copy()
    q = env.model.cam_quat[cam_id]
    w, x, y, z = q
    return {
        "x":    (cam_pos[0] - base_front_x) * 100,
        "y":    cam_pos[1] * 100,
        "z":    cam_pos[2] * 100,
        "pitch": np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1))),
        "yaw":   np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))),
        "roll":  np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))),
        "fovy":  float(env.model.cam_fovy[cam_id]),
    }


def build_obs_from_real(env: LiftCubeProjectionEnv, real_corners_norm, joint_pos: dict) -> np.ndarray | None:
    """Build obs from real camera corners + sim fingertip projections + gripper.

    Returns 13-dim obs for base env or 16-dim for t1v2 env (appends
    gripper_spread, min_gripper_frac, stage_val).
    """
    if real_corners_norm is None:
        return None

    # Fingertips from sim (mirrored joint positions)
    K, R, t = env._get_camera_matrices()
    static_pos = env.data.site_xpos[env._static_fingertip_site_id].reshape(1, 3)
    moving_pos = env.data.site_xpos[env._moving_fingertip_site_id].reshape(1, 3)
    sp_norm = env._project_points_norm(static_pos, K, R, t)[0]
    mp_norm = env._project_points_norm(moving_pos, K, R, t)[0]
    su, sv = sp_norm
    mu, mv = mp_norm

    # Cube corners from real detection (already normalised [0,1])
    corners_flat = real_corners_norm.flatten()  # (8,)

    # Gripper from real robot — convert degrees to MuJoCo joint radians then normalise
    gripper_deg = joint_pos.get("gripper", 7.852)
    gripper_rad = float(np.radians((gripper_deg - 7.852) / 0.648))
    gripper_lo = float(getattr(env, "_gripper_lo", -0.175))
    gripper_hi = float(getattr(env, "_gripper_hi", 1.745))
    gripper_norm = float(np.clip((gripper_rad - gripper_lo) / (gripper_hi - gripper_lo + 1e-8), 0.0, 1.0))

    base_obs = np.array([su, sv, mu, mv, *corners_flat, gripper_norm], dtype=np.float32)

    # For t1v2: append gripper_spread, min_gripper_frac, stage_val
    if isinstance(env, LiftCubeProjectionT1V2Env):
        gripper_spread = float(np.clip(np.linalg.norm(sp_norm - mp_norm), 0.0, 1.0))
        min_frac = float(getattr(env, "_min_gripper_frac", 0.5))
        from src.envs.lift_cube_projection_t1v2 import _STAGE_VALS
        stage_val = float(_STAGE_VALS.get(getattr(env, "_stage", 0), 0.33))
        obs = np.append(base_obs, [gripper_spread, min_frac, stage_val])
    else:
        obs = base_obs

    return np.clip(obs, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _find_latest_run(tier: int) -> tuple[str | None, str | None, bool]:
    """Return (model_path, normalize_path, is_t1v2) for the latest t{tier} run.

    Checks t{tier}v2 pattern first, then legacy t{tier} pattern.
    """
    for pattern in [
        f"lift_proj_t{tier}v2_*/*/best_model/best_model.zip",
        f"lift_proj_t{tier}_*/*/best_model/best_model.zip",
    ]:
        zips = sorted(Path("runs").glob(pattern), key=lambda p: p.stat().st_mtime)
        if zips:
            latest = zips[-1]
            norm = latest.parent.parent / "vec_normalize.pkl"
            is_v2 = "v2" in latest.parts[-4]
            print(f"T{tier}: auto-detected {latest}  (t{tier}{'v2' if is_v2 else ''})")
            return str(latest), str(norm) if norm.exists() else None, is_v2
    return None, None, False


def _load_policy(model_path: str | None, normalize_path: str | None, is_t1v2: bool = False):
    """Load a PPO policy and optional VecNormalize. Returns (policy, vec_norm) or (None, None)."""
    if not model_path:
        return None, None
    if is_t1v2:
        dummy_env = LiftCubeProjectionT1V2Env(curriculum_stage=3, lock_wrist=True,
                                               cam_pos_noise=0.0, cam_euler_noise=0.0)
    else:
        dummy_env = LiftCubeProjectionEnv(curriculum_stage=3, lock_wrist=True)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    vec_norm = None
    if normalize_path:
        vec_norm = VecNormalize.load(normalize_path, dummy_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False
    policy = PPO.load(model_path, env=vec_norm or dummy_vec)
    return policy, vec_norm


def main():
    t1_model, t1_norm, t1_v2 = _find_latest_run(1)
    t2_model, t2_norm, t2_v2 = _find_latest_run(2)
    t3_model, t3_norm, t3_v2 = _find_latest_run(3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model1", type=str, default=t1_model, help="T1 policy .zip")
    parser.add_argument("--normalize1", type=str, default=t1_norm, help="T1 vec_normalize.pkl")
    parser.add_argument("--model2", type=str, default=t2_model, help="T2 policy .zip")
    parser.add_argument("--normalize2", type=str, default=t2_norm, help="T2 vec_normalize.pkl")
    parser.add_argument("--model3", type=str, default=t3_model, help="T3 policy .zip")
    parser.add_argument("--normalize3", type=str, default=t3_norm, help="T3 vec_normalize.pkl")
    parser.add_argument("--base-env", action="store_true",
                        help="Force base 13-dim env (for t0 models)")
    parser.add_argument("--reward-version", type=str, default="v_unified_straddle",
                        help="Reward version / stage (e.g. v_unified_zone for t0, v_unified_straddle for t1)")
    parser.add_argument("--action-scale", type=float, default=REAL_ACTION_SCALE,
                        help=f"Action scale for real robot movement (default: {REAL_ACTION_SCALE})")
    parser.add_argument("--cube-xy", type=float, nargs=2, default=[0.10, 0.0], metavar=("X", "Y"),
                        help="Cube position in metres relative to front edge of robot base (default: 0.10 0.0)")
    args = parser.parse_args()
    real_action_scale = args.action_scale

    # Use t1v2 env if the active policy is v2 (16-dim obs), unless --base-env forced
    if t1_v2 and not args.base_env:
        env = LiftCubeProjectionT1V2Env(
            render_mode="rgb_array", curriculum_stage=3, lock_wrist=True,
            cam_pos_noise=0.0, cam_euler_noise=0.0, fov_noise=0.0,
            freeze_cube=False,
            reward_version=args.reward_version,
        )
    else:
        env = LiftCubeProjectionEnv(
            render_mode="rgb_array", curriculum_stage=3, lock_wrist=True,
            cam_pos_noise=0.0, cam_euler_noise=0.0,
            freeze_cube=False,
            reward_version=args.reward_version,
        )
    env.reset()
    env._renderer = None
    env._overview_renderer = None

    base_front_x = get_base_front_x(env)
    cube_world_x = base_front_x + args.cube_xy[0]
    cube_world_y = args.cube_xy[1]
    print(f"Robot base front edge: x={base_front_x:.4f}m")
    print(f"Cube world position: x={cube_world_x:.4f}m  y={cube_world_y:.4f}m  (from --cube-xy {args.cube_xy[0]} {args.cube_xy[1]})")

    # Load camera intrinsics for undistortion
    cam_K, cam_dist = load_camera_intrinsics()

    # Bake saved wrist cam extrinsic offsets into model so renderer and projection agree
    _extrinsics_path = Path("calibration/wrist_cam_extrinsics.json")
    if _extrinsics_path.exists():
        with open(_extrinsics_path) as _f:
            _ex = json.load(_f)
        pos_off   = np.array(_ex["pos_offset"])
        euler_off = np.array(_ex["euler_offset"])
        cam_id = env._wrist_cam_id
        # Apply position offset
        env.model.cam_pos[cam_id] += pos_off
        # Apply euler offset by composing into existing quaternion
        if np.any(euler_off != 0):
            r, p, y = euler_off
            Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
            Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
            Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
            delta_mat = Rz @ Ry @ Rx
            delta_quat = np.zeros(4)
            mujoco.mju_mat2Quat(delta_quat, delta_mat.flatten())
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, env.model.cam_quat[cam_id], delta_quat)
            env.model.cam_quat[cam_id] = new_quat
        # Zero out offsets so projection math doesn't double-apply them
        env._cam_pos_offset   = np.zeros(3)
        env._cam_euler_offset = np.zeros(3)
        mujoco.mj_forward(env.model, env.data)
        print(f"Baked wrist cam offsets from {_extrinsics_path}")
        print(f"  pos={pos_off}  euler={euler_off}")
    else:
        print("No wrist cam extrinsics found — using XML defaults")

    # Load all three policies
    tier_configs = [
        (args.model1, args.normalize1, t1_v2),
        (args.model2, args.normalize2, t2_v2),
        (args.model3, args.normalize3, t3_v2),
    ]
    policies = []
    for i, (m, n, is_v2) in enumerate(tier_configs, start=1):
        p, vn = _load_policy(m, n, is_t1v2=is_v2)
        policies.append((p, vn, is_v2))
        if p:
            print(f"T{i}: loaded {m}  ({'v2' if is_v2 else 'v1'})")
        else:
            print(f"T{i}: no model found")

    active_tier = 0  # 0-indexed; 0=T1, 1=T2, 2=T3
    policy, vec_norm, active_is_v2 = policies[active_tier]

    robot = RealRobot(port=args.port, use_degrees=True)
    robot.connect()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        robot.disconnect()
        return

    policy_running = False
    sim_policy_running = False
    sim_to_real_running = False
    print("Calibration viewer running.")
    print("  Q=quit  P=snapshot  1/2/3=select tier")
    print("  R=toggle real policy  S=toggle sim-only policy  I=toggle sim->real policy")
    print("  G=capture current gripper spread as min_gripper_frac")

    while True:
        ret, real_frame = cap.read()
        if not ret:
            break

        # --- Real: crop, flip, resize ---
        real_frame = undistort_frame(real_frame, cam_K, cam_dist)
        h, w = real_frame.shape[:2]
        crop_x = (w - h) // 2
        real_square = cv2.flip(real_frame[:, crop_x:crop_x + h], 1)
        real_panel = cv2.resize(real_square, (PANEL_W, PANEL_H))

        # --- Detect blue cube ---
        real_corners_px = detect_blue_corners(real_panel)
        draw_quad(real_panel, real_corners_px, (0, 200, 255), "cube")

        # --- Normalise real corners to [0,1] ---
        real_corners_norm = None
        if real_corners_px is not None:
            real_corners_norm = real_corners_px.astype(np.float32) / np.array([PANEL_W, PANEL_H])

        # --- Read real joints (always, for joint grid display) ---
        joint_pos = robot.get_joint_positions()
        sim_joints = lerobot_to_mujoco(joint_pos)

        # --- Mirror joints into sim (only when sim is running freely, not under policy) ---
        if not sim_policy_running and not sim_to_real_running:
            set_sim_joints(env, sim_joints)

        # --- Pin cube at position relative to front of base ---
        pin_cube(env, base_front_x + args.cube_xy[0], args.cube_xy[1])


        # --- Build obs and run policy on real robot ---
        if policy_running and policy is not None:
            obs = build_obs_from_real(env, real_corners_norm, joint_pos)
            if obs is not None:
                obs_input = obs[np.newaxis, :]
                if vec_norm is not None:
                    obs_input = vec_norm.normalize_obs(obs_input)
                action, _ = policy.predict(obs_input, deterministic=True)

                # Scale action down for safety before applying to sim
                safe_action = action[0].copy()
                safe_action[:3] *= real_action_scale / env.action_scale  # scale xyz
                # Sync target EE pos from current sim state before stepping
                env._target_ee_pos = env.ik.get_ee_position().copy()
                # Apply to sim via IK — sim handles XYZ -> joint angles
                env.step(safe_action)

                # Read resulting sim ctrl (joint commands in radians) -> send to real arm
                # ctrl order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
                sim_ctrl = env.data.ctrl.copy()
                real_cmd = {
                    "shoulder_pan":  float((np.degrees(sim_ctrl[0]) - 22.423) / 1.02857),
                    "shoulder_lift": float(np.degrees(sim_ctrl[1])),
                    "elbow_flex":    float(np.degrees(sim_ctrl[2])),
                    "wrist_flex":    float(np.degrees(sim_ctrl[3])),
                    "wrist_roll":    float(np.degrees(-(sim_ctrl[4] - 1.679))),
                    "gripper":       float(np.clip(0.648 * get_gripper_qpos_deg(env) + 7.852, 0, 60)),
                }
                robot.set_joint_positions(real_cmd)  # use_degrees=True so values are in degrees
                cv2.putText(real_panel, f"POLICY RUNNING  act={action[0,:3]}",
                            (8, PANEL_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # --- Run policy using sim obs, send result to real arm ---
        elif sim_to_real_running and policy is not None:
            obs_sim = env._get_obs()
            obs_input = obs_sim[np.newaxis, :]
            if vec_norm is not None:
                obs_input = vec_norm.normalize_obs(obs_input)
            action, _ = policy.predict(obs_input, deterministic=True)
            safe_action = action[0].copy()
            safe_action[:3] *= real_action_scale / env.action_scale
            env._target_ee_pos = env.ik.get_ee_position().copy()
            env.step(safe_action)
            sim_ctrl = env.data.ctrl.copy()
            real_cmd = {
                "shoulder_pan":  float((np.degrees(sim_ctrl[0]) - 22.423) / 1.02857),
                "shoulder_lift": float(np.degrees(sim_ctrl[1])),
                "elbow_flex":    float(np.degrees(sim_ctrl[2])),
                "wrist_flex":    float(np.degrees(sim_ctrl[3])),
                "wrist_roll":    float(np.degrees(-(sim_ctrl[4] - 1.679))),
                "gripper":       float(np.clip(0.648 * get_gripper_qpos_deg(env) + 7.852, 0, 60)),
            }
            robot.set_joint_positions(real_cmd)
            cv2.putText(real_panel, f"SIM->REAL  act={action[0,:3]}",
                        (8, PANEL_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)

        # --- Run policy in sim only (no real arm commands) ---
        elif sim_policy_running and policy is not None:
            obs_sim = env._get_obs()
            obs_input = obs_sim[np.newaxis, :]
            if vec_norm is not None:
                obs_input = vec_norm.normalize_obs(obs_input)
            action, _ = policy.predict(obs_input, deterministic=True)
            env._target_ee_pos = env.ik.get_ee_position().copy()
            env.step(action[0])
            cv2.putText(real_panel, f"SIM POLICY  act={action[0,:3]}",
                        (8, PANEL_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # --- Sim wrist view with obs overlay ---
        sim_panel = render_sim(env)
        obs_sim = env._get_obs()
        sim_panel = draw_obs_overlay(sim_panel, obs_sim)

        # --- Overview ---
        overview_panel = render_sim_overview(env)

        # --- Joint grid (top-left of real panel) ---
        if sim_policy_running or sim_to_real_running:
            sim_qpos = read_sim_joint_angles(env)
        else:
            sim_qpos = sim_joints

        # --- Build header bars (drawn above each panel) ---
        HEADER_H = 16 + max(len(JOINT_NAMES), 8) * 16 + 8
        cam = get_cam_info(env, base_front_x)

        # Real header: joint table
        real_hdr = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(real_hdr, "REAL", (PANEL_W // 2 - 20, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(real_hdr, f"{'JOINT':<12} {'REAL°':>8} {'SIM°':>7}", (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        for i, name in enumerate(JOINT_NAMES):
            lr_val = joint_pos.get(name, 0.0)
            sim_val_deg = float(np.degrees(sim_qpos.get(name, 0.0)))
            cv2.putText(real_hdr, f"{name[:12]:<12} {lr_val:>8.1f} {sim_val_deg:>7.1f}",
                        (4, 44 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Sim wrist header: cam info
        sim_hdr = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(sim_hdr, "SIM (wrist)", (PANEL_W // 2 - 45, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cam_lines = [
            f"x={cam['x']:+6.2f}cm (fwd/back)   pitch={cam['pitch']:+7.2f}deg (nod up/down)",
            f"y={cam['y']:+6.2f}cm (left/right)  yaw  ={cam['yaw']:+7.2f}deg (turn left/right)",
            f"z={cam['z']:+6.2f}cm (up/down)     roll ={cam['roll']:+7.2f}deg (tilt left/right)",
            f"fovy={cam['fovy']:.2f}deg (zoom — lower=bigger)",
        ]
        for i, line in enumerate(cam_lines):
            cv2.putText(sim_hdr, line, (4, 30 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

        # Overview header: policy status + cube position
        ov_hdr = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(ov_hdr, "SIM (overview)", (PANEL_W // 2 - 55, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tier_label = f"T{active_tier + 1}" if policy else "NO MODEL"
        tier_color = [(0, 0, 255), (0, 165, 255), (0, 255, 180)][active_tier]
        if policy_running:
            status = f"● {tier_label} POLICY ON (REAL)"
        elif sim_policy_running:
            status = f"● {tier_label} POLICY ON (SIM)"
        elif sim_to_real_running:
            status = f"● {tier_label} POLICY ON (SIM->REAL)"
        else:
            status = f"[ {tier_label} selected ]"
        cv2.putText(ov_hdr, status, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tier_color, 1)
        cv2.putText(ov_hdr, f"CUBE  {args.cube_xy[0]*100:.0f}cm fwd  {args.cube_xy[1]*100:.0f}cm side (from base front)",
                    (4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        divider = np.full((PANEL_H, 4, 3), 80, dtype=np.uint8)
        hdr_divider = np.full((HEADER_H, 4, 3), 80, dtype=np.uint8)
        panels     = np.hstack([real_panel,  divider,     sim_panel,  divider,     overview_panel])
        headers    = np.hstack([real_hdr,    hdr_divider, sim_hdr,    hdr_divider, ov_hdr])
        combined   = np.vstack([headers, panels])
        cv2.imshow("Calibration Viewer  |  Real  |  Sim wrist  |  Sim overview", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            fname = f"calibration_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved {fname}")
        elif key == ord('r') and policy is not None:
            policy_running = not policy_running
            sim_policy_running = False
            if policy_running:
                robot.enable_torque()
                env.reset()
                print("Policy RUNNING on real robot — torque enabled")
            else:
                robot.disable_torque()
                print("Policy STOPPED — torque disabled")
        elif key == ord('s') and policy is not None:
            sim_policy_running = not sim_policy_running
            policy_running = False
            sim_to_real_running = False
            if sim_policy_running:
                robot.disable_torque()
                env.reset()
                print("Policy RUNNING in sim only — real arm free")
            else:
                print("Sim policy STOPPED")
        elif key in (ord('1'), ord('2'), ord('3')):
            new_tier = key - ord('1')
            if new_tier != active_tier:
                active_tier = new_tier
                policy, vec_norm, active_is_v2 = policies[active_tier]
                # Stop any running policy when switching
                policy_running = False
                sim_policy_running = False
                sim_to_real_running = False
                robot.disable_torque()
                print(f"Switched to T{active_tier + 1} — {'loaded' if policy else 'NO MODEL'} ({'v2' if active_is_v2 else 'v1'})")
        elif key == ord('g') and isinstance(env, LiftCubeProjectionT1V2Env):
            # Capture current sim gripper state as the ideal min_gripper_frac
            gripper_state = env._get_gripper_state()
            lo, hi = env._gripper_lo, env._gripper_hi
            frac = float(np.clip((gripper_state - lo) / (hi - lo + 1e-8), 0.0, 1.0))
            env._min_gripper_frac = frac
            print(f"Captured min_gripper_frac = {frac:.3f}  (gripper_state={gripper_state:.4f})")
        elif key == ord('i') and policy is not None:
            sim_to_real_running = not sim_to_real_running
            policy_running = False
            sim_policy_running = False
            if sim_to_real_running:
                robot.enable_torque()
                env.reset()
                print("Policy RUNNING sim->real — sim obs drives real arm")
            else:
                robot.disable_torque()
                print("Sim->real policy STOPPED — torque disabled")

    cap.release()
    robot.disconnect()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
