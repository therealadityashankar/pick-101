"""Interactive wrist cam extrinsic tuner.

Side-by-side: real wrist cam (left) vs sim wrist cam (right).
Adjust the sim camera pose with keyboard until they match, then save.

Offsets are saved relative to the XML nominal values — the XML is not
modified. calibration_viewer.py loads calibration/wrist_cam_extrinsics.json
and applies the offsets at startup.

Saved to: calibration/wrist_cam_extrinsics.json

Controls:
    W / S      — translate cam +/- X  (forward / back)
    A / D      — translate cam +/- Y  (left / right)
    R / F      — translate cam +/- Z  (up / down)
    I / K      — pitch  +/-
    J / L      — yaw    +/-
    U / O      — roll   +/-
    SHIFT+key  — 10x coarser step
    SPACE      — reset all offsets to zero
    ENTER      — save to JSON and quit
    ESC / Q    — quit without saving
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.robot.real_robot import RealRobot, JOINT_NAMES
from calibration_viewer import load_camera_intrinsics, undistort_frame
import mujoco

SAVE_PATH = Path("calibration/wrist_cam_extrinsics.json")

TRANS_STEP = 0.0005   # metres  (fine)
ROT_STEP   = 0.002    # radians (fine)
COARSE_MUL = 10       # multiplier when shift is held

PANEL_W = 480
PANEL_H = 480


def get_base_front_x(env: LiftCubeProjectionEnv) -> float:
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    max_x = 0.0
    for i in range(env.model.ngeom):
        if env.model.geom_bodyid[i] == base_id:
            gx = env.model.geom_pos[i, 0]
            hx = env.model.geom_aabb[i, 3]
            max_x = max(max_x, gx + hx)
    return max_x


def pin_cube(env: LiftCubeProjectionEnv, x: float, y: float, z: float = 0.01):
    cube_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    addr = env.model.jnt_qposadr[cube_joint_id]
    env.data.qpos[addr:addr + 3] = [x, y, z]
    env.data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]
    env.data.qvel[env.model.jnt_dofadr[cube_joint_id]:env.model.jnt_dofadr[cube_joint_id] + 6] = 0
    mujoco.mj_forward(env.model, env.data)


def load_existing() -> dict:
    if SAVE_PATH.exists():
        with open(SAVE_PATH) as f:
            return json.load(f)
    return {"pos_offset": [0.0, 0.0, 0.0], "euler_offset": [0.0, 0.0, 0.0]}


def render_sim_wrist(env: LiftCubeProjectionEnv, pos_offset: np.ndarray, euler_offset: np.ndarray) -> np.ndarray:
    """Render wrist cam with offsets applied directly to model.cam_pos/quat, then restored."""
    cam_id = env._wrist_cam_id

    # Save originals
    orig_pos  = env.model.cam_pos[cam_id].copy()
    orig_quat = env.model.cam_quat[cam_id].copy()

    # Apply position offset
    env.model.cam_pos[cam_id] = orig_pos + pos_offset

    # Apply euler offset by composing rotation into the existing quaternion
    if np.any(euler_offset != 0):
        r, p, y = euler_offset
        # Build delta rotation matrix from euler offsets (intrinsic XYZ)
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        delta_mat = Rz @ Ry @ Rx
        # Convert delta rotation matrix to quaternion
        delta_quat = np.zeros(4)
        mujoco.mju_mat2Quat(delta_quat, delta_mat.flatten())
        # Compose with existing quaternion
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, orig_quat, delta_quat)
        env.model.cam_quat[cam_id] = new_quat

    mujoco.mj_forward(env.model, env.data)

    if env._renderer is None:
        env._renderer = mujoco.Renderer(env.model, height=env.CAM_H, width=env.CAM_W)
    env._renderer.update_scene(env.data, camera="wrist_cam")
    img = env._renderer.render()

    # Restore originals
    env.model.cam_pos[cam_id]  = orig_pos
    env.model.cam_quat[cam_id] = orig_quat
    mujoco.mj_forward(env.model, env.data)

    crop_x = (env.CAM_W - env.CAM_H) // 2
    img = img[:, crop_x:crop_x + env.CAM_H]
    img = cv2.resize(img, (PANEL_W, PANEL_H))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_info(img: np.ndarray, pos: np.ndarray, euler: np.ndarray, coarse: bool):
    step_label = f"step x{COARSE_MUL} (COARSE)" if coarse else "step x1 (fine)"
    lines = [
        f"pos   dx={pos[0]:+.4f}m  dy={pos[1]:+.4f}m  dz={pos[2]:+.4f}m",
        f"euler  p={euler[0]:+.4f}   y={euler[1]:+.4f}   r={euler[2]:+.4f}  rad",
        f"{step_label}   SHIFT=coarse   SPACE=reset",
        "ENTER=save  ESC/Q=quit",
    ]
    cv2.rectangle(img, (0, 0), (img.shape[1], 10 + 18 * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, 14 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port for real robot")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--cube-xy", type=float, nargs=2, default=[0.20, 0.0], metavar=("X", "Y"),
                        help="Cube position in metres relative to front edge of robot base (default: 0.20 0.0)")
    args = parser.parse_args()

    SAVE_PATH.parent.mkdir(exist_ok=True)

    existing = load_existing()
    pos_offset   = np.array(existing["pos_offset"],   dtype=np.float64)
    euler_offset = np.array(existing["euler_offset"], dtype=np.float64)

    env = LiftCubeProjectionEnv(
        render_mode="rgb_array", curriculum_stage=3, lock_wrist=True,
        cam_pos_noise=0.0, cam_euler_noise=0.0,
    )
    env.reset()
    env._renderer = None

    base_front_x = get_base_front_x(env)
    cube_x = base_front_x + args.cube_xy[0]
    cube_y = args.cube_xy[1]
    pin_cube(env, cube_x, cube_y)
    print(f"Cube pinned at {args.cube_xy[0]*100:.0f}cm forward from base front (world x={cube_x:.4f})")

    cam_K, cam_dist = load_camera_intrinsics()

    robot = RealRobot(port=args.port, use_degrees=True)
    robot.connect()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        robot.disconnect()
        env.close()
        return

    print("Wrist cam tuner running.")
    if SAVE_PATH.exists():
        print(f"Loaded existing offsets from {SAVE_PATH}")
    print(f"  pos={pos_offset.tolist()}  euler={euler_offset.tolist()}")

    # Key -> (array, index, direction)  for lowercase keys
    KEY_MAP = {
        ord('w'): (pos_offset,   0, +1),
        ord('s'): (pos_offset,   0, -1),
        ord('a'): (pos_offset,   1, -1),
        ord('d'): (pos_offset,   1, +1),
        ord('r'): (pos_offset,   2, +1),
        ord('f'): (pos_offset,   2, -1),
        ord('i'): (euler_offset, 0, +1),
        ord('k'): (euler_offset, 0, -1),
        ord('j'): (euler_offset, 1, +1),
        ord('l'): (euler_offset, 1, -1),
        ord('u'): (euler_offset, 2, +1),
        ord('o'): (euler_offset, 2, -1),
    }
    # Uppercase variants (shift held) map to the same entries
    SHIFT_MAP = {k - 32: v for k, v in KEY_MAP.items() if k >= ord('a')}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = undistort_frame(frame, cam_K, cam_dist)
        h, w = frame.shape[:2]
        crop_x = (w - h) // 2
        real_panel = cv2.flip(frame[:, crop_x:crop_x + h], 1)
        real_panel = cv2.resize(real_panel, (PANEL_W, PANEL_H))

        # Mirror real robot joints into sim
        joint_pos = robot.get_joint_positions()
        scale = np.pi / 180.0
        sim_joints = {
            "shoulder_pan":  (joint_pos.get("shoulder_pan",  0.0) * 1.02857 + 22.423) * scale,
            "shoulder_lift": joint_pos.get("shoulder_lift", 0.0) * scale,
            "elbow_flex":    joint_pos.get("elbow_flex",    0.0) * scale,
            "wrist_flex":    joint_pos.get("wrist_flex",    0.0) * scale,
            "wrist_roll":    -joint_pos.get("wrist_roll",   0.0) * scale + 1.679,
            "gripper":       np.radians((joint_pos.get("gripper", 0.0) - 7.852) / 0.648),
        }
        for name, angle in sim_joints.items():
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                env.data.qpos[env.model.jnt_qposadr[jid]] = angle
        mujoco.mj_forward(env.model, env.data)

        pin_cube(env, cube_x, cube_y)
        sim_panel = render_sim_wrist(env, pos_offset, euler_offset)

        cv2.putText(real_panel, "REAL", (PANEL_W // 2 - 20, PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(sim_panel, "SIM", (PANEL_W // 2 - 15, PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Determine if shift is being held (uppercase key in map)
        # We preview coarse mode in the label based on last key press stored below
        draw_info(sim_panel, pos_offset, euler_offset, coarse=False)

        divider = np.full((PANEL_H, 4, 3), 80, dtype=np.uint8)
        combined = np.hstack([real_panel, divider, sim_panel])
        cv2.imshow("Wrist Cam Tuner  |  Real  |  Sim", combined)

        key = cv2.waitKey(16) & 0xFF
        if key == 0xFF:
            continue

        if key in (ord('q'), 27):
            print("Quit without saving.")
            break

        elif key == 13:   # ENTER
            data = {
                "pos_offset":   pos_offset.tolist(),
                "euler_offset": euler_offset.tolist(),
                "saved_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(SAVE_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {SAVE_PATH}")
            print(f"  pos_offset:   {pos_offset.tolist()}")
            print(f"  euler_offset: {euler_offset.tolist()}")
            break

        elif key == ord(' '):
            pos_offset[:]   = 0.0
            euler_offset[:] = 0.0
            print("Reset to zero.")

        elif key in KEY_MAP:
            arr, idx, sign = KEY_MAP[key]
            step = ROT_STEP if arr is euler_offset else TRANS_STEP
            arr[idx] += sign * step

        elif key in SHIFT_MAP:
            arr, idx, sign = SHIFT_MAP[key]
            step = ROT_STEP if arr is euler_offset else TRANS_STEP
            arr[idx] += sign * step * COARSE_MUL

    cap.release()
    robot.disconnect()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
