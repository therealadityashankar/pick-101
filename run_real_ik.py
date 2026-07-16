"""run_real_ik.py — IK-based pick-and-place on the real SO-101.

Architecture:
  1. Overhead camera detects Jenga block ArUco tag → board mm position
  2. Perspective correction → true interior position
  3. State machine (WAIT → APPROACH → DESCEND → GRASP → LIFT → DONE)
     drives an IK controller each step
  4. Real robot joint positions are synced into sim for accurate Jacobians
  5. 4-panel video is recorded

Usage:
    uv run python run_real_ik.py --port /dev/tty.usbmodem5A680089441
    uv run python run_real_ik.py --no-robot --camera 0
"""

import argparse
import atexit
import tempfile
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import cv2.aruco as aruco
import mujoco
import numpy as np

from src.controllers.ik_controller import IKController
from src.envs.lift_cube import LiftCubeCartesianEnv
from src.robot.joint_calibration import JointCalibration

# ── ArUco detection (must match visualize_irl_block.py) ───────────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.errorCorrectionRate       = 0.7

BORDERED_IDS = set(range(100, 200))
MARGIN_FRAC  = 48 / 200

SQUARE_MM       = 180.0
TAG_MM          = 16.0
GAP_MM          = 2.0
N_OUTER         = 10
N_INNER         = 8
INSET_MM        = 2 * TAG_MM + GAP_MM   # 34mm
BOARD_OFFSET_MM = 70.0   # distance from board bottom edge to robot base tip

OUT_W = 600
OUT_H = 600

# ── Linear perspective correction (from tests/test_aruco_homography_3d.py) ───
CORR_Sx =  0.95596
CORR_Bx =    3.644
CORR_Sy =  0.93985
CORR_By =   37.645

# ── Coordinate mapping ────────────────────────────────────────────────────────
ROBOT_INT_X      = 56.0    # interior mm — lateral centre of robot base
ROBOT_INT_Y      = (SQUARE_MM - INSET_MM) + BOARD_OFFSET_MM   # 216mm
ROBOT_BASE_TIP_X = 0.055   # sim metres — front face of robot base from world origin
CUBE_HALF        = 0.015   # sim metres — half the cube side
CUBE_HALF_MM     = CUBE_HALF * 1000

# Real jenga block: 75 × 25 × 15 mm. Long axis lies along interior Y = sim X.
BLOCK_HALF_X = 0.0375   # sim metres — half length  (75mm, along sim x)
BLOCK_HALF_Y = 0.0125   # sim metres — half width   (25mm, along sim y)
BLOCK_HALF_Z = 0.0075   # sim metres — half height  (15mm)


def make_block_jenga(model):
    """Resize the cube geom in a loaded model to the real jenga block dims."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if gid != -1:
        model.geom_size[gid] = [BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z]

# ── IK / grasp parameters ─────────────────────────────────────────────────────
HOVER_Z       = 0.10    # m above board to hover before descending
GRASP_Z       = 0.022   # m above board to close gripper (just below block top)
LIFT_Z        = 0.12    # m above board for final lift
IK_Z_OFFSET   = 0.1     # m — global vertical offset added to every IK target
                        # (positive = aim higher; compensates real-arm sag /
                        #  calibration error between sim and real)
POS_THRESHOLD = 0.0175   # m — how close EE must be before state transition
                        # (real arm plateaus ~15mm short due to damped IK +
                        #  real→sim sync; threshold must sit above that)
GRASP_STEPS   = 20      # steps to hold while closing gripper
IK_GAIN       = 0.8     # proportional gain for IK steps
HOLD_AFTER_DONE_S = 5.0   # keep torque on and hold pose this long after DONE

# Render-only wrist_roll offset (radians) added to the VIZ panels. Cosmetic fix
# for the badly-fit wrist_roll joint calibration (R²=0.45) — does NOT affect
# control, IK, or what is sent to the robot.
VIZ_WRIST_ROLL_OFFSET = 1.57   # ≈ +90°; tune by eye

# Suffix appended to recording filenames — update when you change the setup
# so recordings are traceable to the configuration that produced them.
RUN_SUFFIX = "zy_adjusted"

# Absolute wrist_roll (lerobot-normalized) the jaw is set to before the pick
# sequence starts — captured from the physical arm in its known-good pose.
WRIST_ROLL_PICK = -0.27

# ── Placement (runs after the pick; set PLACE_ENABLED = False to skip) ────────
# Absolute sim coordinates: +x away from robot, +y robot's left, +z up.
# Values are loaded per-block from wall_layout.yaml (keyed by --block-tag).
# The dict below is the fallback if no YAML entry exists for the block.
PLACE_ENABLED = True
PLACE = {
    "place_x":         0.22,
    "place_y":         0.06,
    "place_hover_z":   0.15,   # travel height on the way to the drop spot
    "place_release_z": 0.10,   # EE height at which the jaw opens
    "place_open_val":  50.0,   # gripper value for release
    "place_wrist_roll": -0.28, # abs wrist_roll before release (None = untouched)
}


def load_place_config(path, block_tag):
    """Overlay wall_layout.yaml defaults + the block's entry onto PLACE."""
    import yaml
    p = Path(path)
    if not p.exists():
        print(f"[place] {path} not found — using built-in defaults")
        return
    cfg = yaml.safe_load(p.read_text()) or {}
    PLACE.update(cfg.get("defaults", {}))
    blocks = cfg.get("blocks", {})
    entry = blocks.get(block_tag, blocks.get(str(block_tag)))
    if entry is None:
        print(f"[place] no entry for block {block_tag} in {path} — using defaults")
    else:
        PLACE.update(entry)
    print(f"[place] block {block_tag}: "
          f"x={PLACE['place_x']}  y={PLACE['place_y']}  "
          f"release_z={PLACE['place_release_z']}")

def load_sweep_config(path, sweep_id):
    """Return the sweep entry (dict) for --sweep N from wall_layout.yaml."""
    import yaml
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[sweep] {path} not found")
    cfg = yaml.safe_load(p.read_text()) or {}
    sweeps = cfg.get("sweeps", {})
    entry = sweeps.get(sweep_id, sweeps.get(str(sweep_id)))
    if entry is None:
        raise SystemExit(f"[sweep] no sweep {sweep_id} in {path}")
    print(f"[sweep] {sweep_id}: {entry['start']} → {entry['end']}  z={entry['sweep_z']}")
    return entry


def perform_sweep_steps(robot, s):
    """Drag the (closed) gripper from start to end to push/smooth blocks."""
    sx, sy = s["start"]
    ex, ey = s["end"]
    hover_z, sweep_z = s["hover_z"], s["sweep_z"]
    it, dl = s.get("iters", 80), s.get("delay", 0.03)

    print("  sweep: close gripper"); yield from ramp_joint_steps(robot, "gripper", s.get("gripper", 0.0), steps=25, delay=0.01)
    if s.get("wrist_roll") is not None:
        print("  sweep: set wrist_roll"); yield from ramp_joint_steps(robot, "wrist_roll", s["wrist_roll"], steps=25, delay=0.01)
    print("  sweep: travel to start"); yield from move_ee_abs_steps(robot, sx, sy, hover_z, iters=60, delay=0.02, gain=0.8)
    print("  sweep: descend");         yield from move_ee_abs_steps(robot, sx, sy, sweep_z, iters=40, delay=0.02, gain=0.8)
    print("  sweep: drag to end");     yield from move_ee_abs_steps(robot, ex, ey, sweep_z, iters=it, delay=dl, gain=0.6, linear=True)
    # Optional: back off to post_end (still low) before lifting, so the retreat
    # doesn't drag the just-aligned bricks out of place.
    rx, ry = ex, ey
    if s.get("post_end") is not None:
        rx, ry = s["post_end"]
        print("  sweep: back off");    yield from move_ee_abs_steps(robot, rx, ry, sweep_z, iters=40, delay=dl, gain=0.6, linear=True)
    print("  sweep: retreat");         yield from move_ee_abs_steps(robot, rx, ry, hover_z, iters=40, delay=0.02, gain=0.8)
    print("perform_sweep: done")


def perform_sweep(robot, s):
    """Blocking sweep (no recording)."""
    for _ in perform_sweep_steps(robot, s):
        pass


# ── Finish (runs last; return the whole arm to this home pose) ───────────────
# Captured from the physical arm — every run ends back here.
FINISH_ENABLED = True
FINISH_STEPS   = 60      # more steps = slower, smoother return
FINISH_DELAY   = 0.02    # s per micro-step
HOME_POSE = {
    "shoulder_pan":  66.21,
    "shoulder_lift":  4.23,
    "elbow_flex":    17.23,
    "wrist_flex":    65.92,
    "wrist_roll":    -1.00,
    "gripper":         5.19,
}

GRIPPER_OPEN  = -1.0    # maps to ctrlrange min
GRIPPER_CLOSE =  1.0    # maps to ctrlrange max

# ── Joint calibration ─────────────────────────────────────────────────────────
_calibration: JointCalibration | None = None


def get_calibration(sim_ranges) -> JointCalibration:
    global _calibration
    if _calibration is None:
        _calibration = JointCalibration(sim_ranges=sim_ranges)
    return _calibration


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WAIT    = auto()   # no block detected yet
    APPROACH = auto()  # hovering above block, gripper open
    DESCEND  = auto()  # lowering to grasp height
    PICK     = auto()  # navigation done — hand over to perform_pick()
    DONE     = auto()


class StateMachine:
    def __init__(self):
        self.phase        = Phase.WAIT
        self.grasp_count  = 0
        self.block_pos    = None   # last-known sim block centre (x, y, z)

    def update_block(self, block_sim_pos):
        if block_sim_pos is not None:
            self.block_pos = block_sim_pos
        if self.phase == Phase.WAIT and self.block_pos is not None:
            self.phase = Phase.APPROACH
            print("Block detected — APPROACH")

    def tick(self, ee_pos) -> tuple[np.ndarray, float, bool]:
        """Return (target_pos, gripper_action, done)."""
        if self.block_pos is None or self.phase == Phase.WAIT:
            # Hold still at current ee position
            return ee_pos.copy(), GRIPPER_OPEN, False

        bx, by, _ = self.block_pos

        if self.phase == Phase.APPROACH:
            target = np.array([bx, by, HOVER_Z + IK_Z_OFFSET])
            if np.linalg.norm(ee_pos - target) < POS_THRESHOLD:
                self.phase = Phase.DESCEND
                print("Reached hover — DESCEND")
            return target, GRIPPER_OPEN, False

        if self.phase == Phase.DESCEND:
            target = np.array([bx, by, GRASP_Z + IK_Z_OFFSET])
            if np.linalg.norm(ee_pos - target) < POS_THRESHOLD:
                self.phase = Phase.PICK
                print("At grasp position — PICK (handing over to perform_pick)")
            return target, GRIPPER_OPEN, False

        # PICK is executed by perform_pick() in the main loop; DONE afterwards
        return ee_pos.copy(), GRIPPER_OPEN, self.phase == Phase.DONE


# ─────────────────────────────────────────────────────────────────────────────
# Pick maneuver — EDIT THIS RAW
#
# Runs ONCE on the real robot after IK navigation has placed the gripper tip
# at the grasp position. Direct joint commands, blocking. Joint values are
# lerobot-normalized: arm joints −100..100, gripper 0 (closed) .. 100 (open).
# ─────────────────────────────────────────────────────────────────────────────

def ramp_joint_steps(robot, joint, target, steps=50, delay=0.01):
    """Generator: smoothly move a single joint, yielding after each micro-step."""
    current = robot.get_joint_positions()
    start = current[joint]
    for i in range(1, steps + 1):
        t = i / steps
        current[joint] = start + t * (target - start)
        robot.set_joint_positions(current)
        time.sleep(delay)
        yield


def ramp_joint(robot, joint, target, steps=50, delay=0.01):
    """Blocking version (used by pick_tuner)."""
    for _ in ramp_joint_steps(robot, joint, target, steps, delay):
        pass


# ── Cartesian end-effector control (usable in perform_pick and pick_tuner) ───
# move_ee(robot, dx, dy, dz) nudges the real gripper tip by metres in sim
# coordinates: +x = away from robot, +y = robot's left, +z = up.

_EE: dict = {}


def _ee_ctx():
    """Lazily build a private MuJoCo model + IK controller for EE control."""
    if not _EE:
        model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).parent / "models/so101/lift_cube.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ik = IKController(model, data, end_effector_site="graspframe", damping=0.05)
        sim_ranges = model.actuator_ctrlrange[:6].copy()
        _EE.update(model=model, data=data, ik=ik, sim_ranges=sim_ranges,
                   cal=get_calibration(sim_ranges))
    return _EE


def _sync_real_to_sim(robot, c):
    joints = robot.get_joint_positions()
    q = c["cal"].real_to_sim(joints, c["sim_ranges"])
    c["data"].qpos[:6] = q
    c["data"].ctrl[:6] = q
    mujoco.mj_forward(c["model"], c["data"])
    return q


def get_ee_pos(robot):
    """Current real end-effector position in sim coordinates (metres)."""
    c = _ee_ctx()
    _sync_real_to_sim(robot, c)
    return c["ik"].get_ee_position()


EE_TOL = 0.006   # m — stop a move early once the EE is within this of target


def _move_ee_to_steps(robot, target, iters, delay, gain, linear=False):
    """Generator: drive the real EE toward an absolute sim target via IK.

    linear=False: aim at the final target every step → proportional (fast then
    decaying) approach; exits early within EE_TOL.
    linear=True: interpolate the sub-target linearly from the start position to
    the target across all iters → roughly CONSTANT velocity (no decay). Used for
    the sweep drag so the push speed stays even."""
    c = _ee_ctx()
    target = np.asarray(target, dtype=float)
    _sync_real_to_sim(robot, c)
    start = c["ik"].get_ee_position().copy()
    for i in range(iters):
        q = _sync_real_to_sim(robot, c)
        if linear:
            sub = start + (target - start) * ((i + 1) / iters)   # constant-speed waypoint
        else:
            if np.linalg.norm(c["ik"].get_ee_position() - target) < EE_TOL:
                break
            sub = target
        # Preserve the current gripper opening
        lo, hi = c["sim_ranges"][5]
        grip_action = 2 * (q[5] - lo) / (hi - lo) - 1
        ctrl = c["ik"].step_toward_target(sub, gripper_action=grip_action, gain=gain)
        c["data"].ctrl[:] = ctrl
        mujoco.mj_step(c["model"], c["data"])
        real_targets = c["cal"].sim_to_real(c["data"].ctrl[:6].copy(), c["sim_ranges"])
        robot.set_joint_positions(real_targets)
        time.sleep(delay)
        yield
    final = get_ee_pos(robot)
    err = np.linalg.norm(final - target)
    print(f"move_ee: ee=({final[0]:+.3f},{final[1]:+.3f},{final[2]:+.3f})  residual={err*1000:.1f}mm")


def move_ee_steps(robot, dx=0.0, dy=0.0, dz=0.0, iters=60, delay=0.05, gain=0.4):
    """Generator: move the real EE by (dx, dy, dz) metres RELATIVE via IK."""
    c = _ee_ctx()
    _sync_real_to_sim(robot, c)
    target = c["ik"].get_ee_position() + np.array([dx, dy, dz])
    yield from _move_ee_to_steps(robot, target, iters, delay, gain)


def move_ee_abs_steps(robot, x, y, z, iters=80, delay=0.05, gain=0.4, linear=False):
    """Generator: move the real EE to the ABSOLUTE sim position (x, y, z).

    linear=True gives a constant-speed move (no proportional decay)."""
    yield from _move_ee_to_steps(robot, np.array([x, y, z]), iters, delay, gain, linear)


def move_ee(robot, dx=0.0, dy=0.0, dz=0.0, iters=60, delay=0.05, gain=0.4):
    """Blocking relative move (used by pick_tuner). Returns final EE pos."""
    for _ in move_ee_steps(robot, dx, dy, dz, iters, delay, gain):
        pass
    return get_ee_pos(robot)


def move_ee_abs(robot, x, y, z, iters=80, delay=0.05, gain=0.4):
    """Blocking absolute move (used by pick_tuner). Returns final EE pos."""
    for _ in move_ee_abs_steps(robot, x, y, z, iters, delay, gain):
        pass
    return get_ee_pos(robot)


def perform_pick_steps(robot):
    """The actual pick motion as a generator — the main loop advances it while
    recording video frames between micro-steps. Edit the sequence freely; use
    `yield from` for each motion so recording continues throughout."""
    pos = robot.get_joint_positions()
    print(f"perform_pick: wrist_roll={pos['wrist_roll']:+.1f}  gripper={pos['gripper']:.1f}")

    # Save the post-navigation pose so pick_tuner.py can restore it
    import json
    Path(".calibration").mkdir(exist_ok=True)
    with open(".calibration/pick_start_positions.json", "w") as f:
        json.dump(pos, f, indent=2)
    print("Saved pose → .calibration/pick_start_positions.json")

    # ── Pick sequence (tuned with pick_tuner.py) ─────────────────────────────
    print("  pick: square wrist");   yield from ramp_joint_steps(robot, "wrist_roll", WRIST_ROLL_PICK, steps=25, delay=0.01)
    print("  pick: open gripper");   yield from ramp_joint_steps(robot, "gripper", 50, steps=25, delay=0.01)
    print("  pick: approach");       yield from move_ee_steps(robot, dx=0.02, dy=0.016, dz=-0.046, iters=40, delay=0.02, gain=0.8)
    print("  pick: descend");        yield from move_ee_steps(robot, dz=-0.015, iters=30, delay=0.02, gain=0.8)
    print("  pick: close gripper");  yield from ramp_joint_steps(robot, "gripper", 2, steps=30, delay=0.015)
    print("  pick: lift");           yield from move_ee_steps(robot, dz=0.1, iters=40, delay=0.02, gain=0.8)

    print("perform_pick: done")

    if PLACE_ENABLED:
        yield from perform_place_steps(robot)

    if FINISH_ENABLED:
        yield from perform_finish_steps(robot)


def perform_place_steps(robot):
    """Carry the held block to the absolute PLACE position and release it."""
    px, py     = PLACE["place_x"], PLACE["place_y"]
    hover_z     = PLACE["place_hover_z"]
    release_z   = PLACE["place_release_z"]
    open_val    = PLACE["place_open_val"]
    wrist_roll  = PLACE["place_wrist_roll"]
    print(f"perform_place: → ({px:.3f}, {py:.3f})")

    print("  place: travel");   yield from move_ee_abs_steps(robot, px, py, hover_z, iters=50, delay=0.02, gain=0.8)
    if wrist_roll is not None:
        print("  place: square block"); yield from ramp_joint_steps(robot, "wrist_roll", wrist_roll, steps=25, delay=0.01)
    print("  place: descend");  yield from move_ee_abs_steps(robot, px, py, release_z, iters=40, delay=0.02, gain=0.8)
    print("  place: release");  yield from ramp_joint_steps(robot, "gripper", open_val, steps=25, delay=0.01)
    print("  place: retreat");  yield from move_ee_abs_steps(robot, px, py, hover_z, iters=40, delay=0.02, gain=0.8)

    print("perform_place: done")


def perform_finish_steps(robot):
    """Return the whole arm to HOME_POSE at end of operations."""
    print("perform_finish: returning to home pose")
    start = robot.get_joint_positions()
    for i in range(1, FINISH_STEPS + 1):
        t = i / FINISH_STEPS
        interp = {j: start[j] + t * (HOME_POSE[j] - start[j]) for j in HOME_POSE}
        robot.set_joint_positions(interp)
        time.sleep(FINISH_DELAY)
        yield
    print("perform_finish: done")


def perform_pick(robot):
    """Blocking version (no recording) — kept for interactive use."""
    for _ in perform_pick_steps(robot):
        pass


def perform_finish(robot):
    """Blocking version (no recording)."""
    for _ in perform_finish_steps(robot):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Board detection helpers (unchanged from run_real_t1.py)
# ─────────────────────────────────────────────────────────────────────────────

def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    centres = {}
    tag_id  = 0

    def side_centres(inset_mm, n, side):
        half   = tag_mm / 2
        margin = inset_mm + half
        span   = square_mm - 2 * margin
        positions = []
        for i in range(n):
            t     = i / (n - 1) if n > 1 else 0.5
            along = margin + t * span
            if side == "top":
                positions.append((along, inset_mm + half))
            elif side == "bottom":
                positions.append((along, square_mm - inset_mm - half))
            elif side == "left":
                if 0 < i < n - 1:
                    positions.append((inset_mm + half, along))
            elif side == "right":
                if 0 < i < n - 1:
                    positions.append((square_mm - inset_mm - half, along))
        return positions

    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(0, n_outer, side):
            centres[tag_id] = (cx, cy)
            tag_id += 1
    inner_inset = tag_mm + gap_mm
    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(inner_inset, n_inner, side):
            centres[tag_id] = (cx, cy)
            tag_id += 1
    return centres


def compute_homography(corners_dict, tag_centres_mm, square_mm):
    src_pts, dst_pts = [], []
    for tag_id, img_corners in corners_dict.items():
        if tag_id not in tag_centres_mm:
            continue
        src_pts.append([float(img_corners[:, 0].mean()), float(img_corners[:, 1].mean())])
        cx_mm, cy_mm = tag_centres_mm[tag_id]
        dst_pts.append([cx_mm * OUT_W / square_mm, cy_mm * OUT_W / square_mm])
    if len(src_pts) < 4:
        return None, len(src_pts)
    H, mask = cv2.findHomography(
        np.array(src_pts, np.float32), np.array(dst_pts, np.float32), cv2.RANSAC, 5.0)
    return H, int(mask.sum()) if mask is not None else 0


def bordered_corners_image(img_corners):
    tl, tr, br, bl = img_corners.astype(np.float64)
    right = tr - tl; down = bl - tl; m = MARGIN_FRAC
    return np.array([tl - m*right - m*down, tr + m*right - m*down,
                     br + m*right + m*down, bl - m*right + m*down], dtype=np.float64)


def get_interior_pos(img_corners, H, scale, inset_mm):
    tl_img = img_corners[0].astype(np.float32)
    board_pt = cv2.perspectiveTransform(tl_img.reshape(1, 1, 2), H).reshape(2)
    return float(board_pt[0]) / scale - inset_mm, float(board_pt[1]) / scale - inset_mm


def perspective_correct(obs_x, obs_y, h_mm=None):
    return CORR_Sx * obs_x + CORR_Bx, CORR_Sy * obs_y + CORR_By


def interior_to_sim(int_x_mm, int_y_mm):
    # perspective_correct already returns the BLOCK CENTRE in interior mm
    # (the calibration's "true" corner values are block centres), so no
    # further tag-corner→centre offset is needed here.
    sim_x = ROBOT_BASE_TIP_X + (ROBOT_INT_Y - int_y_mm) / 1000.0
    sim_y = (ROBOT_INT_X - int_x_mm) / 1000.0
    return sim_x, sim_y, BLOCK_HALF_Z


# ─────────────────────────────────────────────────────────────────────────────
# Rectified top-down view
# ─────────────────────────────────────────────────────────────────────────────

def draw_rectified(frame, H, square_mm, tag_mm, corners_dict,
                   tag_centres_mm, block_tag, block_height, scale, inset_mm):
    if H is None:
        blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No homography", (20, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return blank

    warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))

    interior_size_mm = square_mm - 2 * inset_mm
    inset_px = int(inset_mm * scale)
    isize_px = int(interior_size_mm * scale)

    cv2.rectangle(warped, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)
    cv2.rectangle(warped, (inset_px, inset_px),
                  (inset_px + isize_px, inset_px + isize_px), (0, 0, 0), 2)

    for d_mm in range(10, int(interior_size_mm), 10):
        d_px  = inset_px + int(d_mm * scale)
        color = (80, 80, 80) if d_mm % 20 == 0 else (160, 160, 160)
        cv2.line(warped, (d_px, inset_px), (d_px, inset_px + isize_px), color, 1)
        cv2.line(warped, (inset_px, d_px), (inset_px + isize_px, d_px), color, 1)

    for tid, (cx_mm, cy_mm) in tag_centres_mm.items():
        cv2.circle(warped, (int(cx_mm * scale), int(cy_mm * scale)), 3, (0, 160, 0), -1)

    if block_tag in corners_dict:
        img_corners = corners_dict[block_tag]
        b_img   = bordered_corners_image(img_corners)
        b_board = cv2.perspectiveTransform(
            b_img.astype(np.float32).reshape(1, -1, 2), H).reshape(-1, 2)
        cv2.polylines(warped, [b_board.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 2)
        tl_board = cv2.perspectiveTransform(
            img_corners[0].astype(np.float32).reshape(1, 1, 2), H).reshape(2)
        tl_px = (int(tl_board[0]), int(tl_board[1]))
        x_int = float(tl_board[0]) / scale - inset_mm
        y_int = float(tl_board[1]) / scale - inset_mm
        cv2.drawMarker(warped, tl_px, (0, 200, 255), cv2.MARKER_CROSS, 14, 2)
        cx_int, cy_int = perspective_correct(x_int, y_int, block_height)
        cx_px = inset_px + int(cx_int * scale)
        cy_px = inset_px + int(cy_int * scale)
        cv2.circle(warped, (cx_px, cy_px), 7, (0, 255, 0), -1)
        cv2.putText(warped, f"({cx_int:.1f},{cy_int:.1f})mm",
                    (cx_px + 8, cy_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return warped


# ─────────────────────────────────────────────────────────────────────────────
# Board scene builder (for visualization env)
# ─────────────────────────────────────────────────────────────────────────────

def board_to_sim(board_x_mm, board_y_mm):
    # Board bottom edge is BOARD_OFFSET_MM in front of the robot tip in sim
    sim_x = ROBOT_BASE_TIP_X + (SQUARE_MM + BOARD_OFFSET_MM - board_y_mm) / 1000.0
    sim_y = -(board_x_mm - SQUARE_MM / 2) / 1000.0
    return sim_x, sim_y


def build_scene_xml(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    tag_half  = tag_mm / 2 / 1000
    board_h   = square_mm / 2 / 1000
    inset_mm  = 2 * tag_mm + gap_mm
    int_half  = (square_mm / 2 - inset_mm) / 1000
    board_cx, board_cy = board_to_sim(square_mm / 2, square_mm / 2)

    geoms = []
    geoms.append(
        f'<geom name="board_surface" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} -0.0004" '
        f'size="{board_h:.4f} {board_h:.4f} 0.0004" '
        f'rgba="0.85 0.80 0.65 1" contype="0" conaffinity="0" group="1"/>'
    )
    geoms.append(
        f'<geom name="board_interior" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} 0.00005" '
        f'size="{int_half:.4f} {int_half:.4f} 0.00005" '
        f'rgba="0.92 0.88 0.72 1" contype="0" conaffinity="0" group="1"/>'
    )
    tag_centres = board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner)
    for i, (bx, by) in tag_centres.items():
        sx, sy = board_to_sim(bx, by)
        geoms.append(
            f'<geom name="btag_{i}" type="box" '
            f'pos="{sx:.4f} {sy:.4f} 0.0001" '
            f'size="{tag_half:.4f} {tag_half:.4f} 0.0001" '
            f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
        )
    geoms.append(
        f'<geom name="base_tip_line" type="box" '
        f'pos="{ROBOT_BASE_TIP_X:.4f} 0.0000 0.0003" '
        f'size="0.0008 {board_h:.4f} 0.0003" '
        f'rgba="1 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
    )
    geoms_xml = "\n        ".join(geoms)
    topdown_cam = (
        f'<camera name="topdown" pos="{board_cx:.4f} {board_cy:.4f} 0.6" '
        f'euler="0 0 3.14159" fovy="30"/>'
    )
    base_xml = Path(__file__).parent / "models/so101/lift_cube.xml"
    xml = f"""<mujoco model="lift_cube_board">
    <include file="{base_xml}"/>
    <worldbody>
        {geoms_xml}
        {topdown_cam}
    </worldbody>
</mujoco>"""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False,
        dir=str(Path(__file__).parent / "models/so101")
    )
    tmp.write(xml.encode())
    tmp.close()
    atexit.register(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",         type=str,   default=None)
    parser.add_argument("--camera",       type=int,   default=0)
    parser.add_argument("--camera2",      type=int,   default=None,
                        help="Optional second camera index — raw stream saved to recordings/camera2/")
    parser.add_argument("--block-tag",    type=int,   default=101)
    parser.add_argument("--block-height", type=float, default=15.0,
                        help="Height of tag above board in mm (cube top = 30mm)")
    parser.add_argument("--wall-layout",  type=str,   default="wall_layout.yaml",
                        help="YAML of per-block placement targets (keyed by block tag)")
    parser.add_argument("--sweep",        type=int,   default=None,
                        help="Run a sweep (push/smooth blocks) instead of a pick; "
                             "N selects the sweep entry in wall_layout.yaml")
    parser.add_argument("--no-robot",     action="store_true")
    parser.add_argument("--output",       type=str,   default="real_ik_run.mp4")
    parser.add_argument("--fps",          type=float, default=10.0)
    parser.add_argument("--max-steps",    type=int,   default=1000)
    args = parser.parse_args()

    if args.port is None and not args.no_robot:
        parser.error("Provide --port or use --no-robot")

    # ── Sweep mode: standalone, no camera/detection/pick ─────────────────────
    if args.sweep is not None:
        sweep_cfg = load_sweep_config(args.wall_layout, args.sweep)
        if args.no_robot:
            print("(no robot — nothing to sweep)")
            return
        from src.robot.real_robot import RealRobot
        robot = RealRobot(port=args.port)
        robot.connect()
        robot.enable_torque()

        # Raw camera recording (same folders as pick runs)
        cap = cv2.VideoCapture(args.camera)
        cap2 = None
        if args.camera2 is not None:
            cap2 = cv2.VideoCapture(args.camera2)
            if not cap2.isOpened():
                cap2 = None
        run_stamp   = time.strftime("%Y%m%d_%H%M%S") + f"_sweep{args.sweep}_" + RUN_SUFFIX
        fourcc      = cv2.VideoWriter_fourcc(*"avc1")
        cam_writer  = cam2_writer = None

        def record_frame():
            nonlocal cam_writer, cam2_writer
            ok, frame = cap.read()
            if ok:
                if cam_writer is None:
                    Path("recordings/camera1").mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    cam_writer = cv2.VideoWriter(
                        f"recordings/camera1/{run_stamp}.mp4", fourcc, args.fps, (w, h))
                cam_writer.write(frame)
            if cap2 is not None:
                ok2, frame2 = cap2.read()
                if ok2:
                    if cam2_writer is None:
                        Path("recordings/camera2").mkdir(parents=True, exist_ok=True)
                        h2, w2 = frame2.shape[:2]
                        cam2_writer = cv2.VideoWriter(
                            f"recordings/camera2/{run_stamp}.mp4", fourcc, args.fps, (w2, h2))
                    cam2_writer.write(frame2)

        try:
            steps = perform_sweep_steps(robot, sweep_cfg)
            if FINISH_ENABLED:
                import itertools
                steps = itertools.chain(steps, perform_finish_steps(robot))
            for _ in steps:
                record_frame()
        finally:
            time.sleep(1.0)
            record_frame()
            if cam_writer is not None:
                cam_writer.release()
                print(f"Saved recordings/camera1/{run_stamp}.mp4")
            if cam2_writer is not None:
                cam2_writer.release()
            cap.release()
            if cap2 is not None:
                cap2.release()
            robot.disable_torque()
            robot.disconnect()
        return

    # Load this block's placement target from the wall layout
    load_place_config(args.wall_layout, args.block_tag)

    scale       = OUT_W / SQUARE_MM
    inset_mm    = INSET_MM
    tag_centres_mm = board_tag_centres(SQUARE_MM, TAG_MM, GAP_MM, N_OUTER, N_INNER)

    # ── IK model (bare lift_cube.xml — robot only, no board clutter) ──────────
    print("Loading IK model...")
    ik_model_path = Path(__file__).parent / "models/so101/lift_cube.xml"
    ik_model = mujoco.MjModel.from_xml_path(str(ik_model_path))
    ik_data  = mujoco.MjData(ik_model)
    mujoco.mj_resetData(ik_model, ik_data)
    mujoco.mj_forward(ik_model, ik_data)

    ik = IKController(ik_model, ik_data, end_effector_site="graspframe", damping=0.05)
    sim_ranges = ik_model.actuator_ctrlrange[:6].copy()
    cal        = get_calibration(sim_ranges)

    # ── Visualization env (board scene, for side-view and top-down panels) ────
    print("Building board scene...")
    scene_xml = build_scene_xml(SQUARE_MM, TAG_MM, GAP_MM, N_OUTER, N_INNER)
    viz_env = LiftCubeCartesianEnv(render_mode="rgb_array", scene_path=scene_xml)
    viz_env.reset()
    make_block_jenga(viz_env.model)
    make_block_jenga(ik_model)

    viz_cube_jnt_id   = mujoco.mj_name2id(viz_env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    viz_cube_qpos_adr = viz_env.model.jnt_qposadr[viz_cube_jnt_id]

    # Park cube below floor until first detection
    viz_env.data.qpos[viz_cube_qpos_adr:viz_cube_qpos_adr + 3] = [0.0, 0.0, -1.0]
    viz_env.data.qpos[viz_cube_qpos_adr + 3:viz_cube_qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(viz_env.model, viz_env.data)

    if viz_env._renderer is None:
        viz_env._renderer = mujoco.Renderer(viz_env.model, height=480, width=640)

    # ── Connect to real robot ─────────────────────────────────────────────────
    robot = None
    if not args.no_robot:
        from src.robot.real_robot import RealRobot
        robot = RealRobot(port=args.port)
        robot.connect()
        robot.enable_torque()
        print("Connected to real robot")

        # Seed IK model with current real joint positions
        real_joints = robot.get_joint_positions()
        init_ctrl   = cal.real_to_sim(real_joints, sim_ranges)
        ik_data.ctrl[:6] = init_ctrl
        ik_data.qpos[:6] = init_ctrl
        mujoco.mj_forward(ik_model, ik_data)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    cap2 = None
    if args.camera2 is not None:
        cap2 = cv2.VideoCapture(args.camera2)
        if not cap2.isOpened():
            print(f"Warning: failed to open camera2 ({args.camera2}) — continuing without it")
            cap2 = None

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    sm            = StateMachine()
    writer        = None
    cam_writer    = None
    cam2_writer   = None
    frame_count   = 0
    block_sim_pos = None
    H             = None
    n_inliers     = 0
    cx_int = cy_int = 0.0
    detected_last = False
    pick_gen      = None   # perform_pick_steps generator, created on PICK
    done_time     = None   # when DONE was reached (for the recorded hold)

    print(f"Running IK pick-and-place for up to {args.max_steps} steps. Recording → {args.output}")

    try:
        for step in range(args.max_steps):
            t_start = time.time()

            # ── Detect block ──────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                break

            frame2 = None
            if cap2 is not None:
                ret2, frame2 = cap2.read()
                if not ret2:
                    frame2 = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det_corners, det_ids, _ = detector.detectMarkers(gray)
            corners_dict = {}
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    corners_dict[int(tid)] = det_corners[i].reshape(4, 2)

            H, n_inliers = compute_homography(corners_dict, tag_centres_mm, SQUARE_MM)

            if H is not None and args.block_tag in corners_dict:
                img_corners = corners_dict[args.block_tag]
                x_int, y_int = get_interior_pos(img_corners, H, scale, inset_mm)
                cx_int, cy_int = perspective_correct(x_int, y_int, args.block_height)
                block_sim_pos = interior_to_sim(cx_int, cy_int)
                if not detected_last:
                    print(f"[{step:4d}] Block detected  "
                          f"int=({cx_int:.1f},{cy_int:.1f})mm  "
                          f"sim=({block_sim_pos[0]:.3f},{block_sim_pos[1]:.3f})m")
                detected_last = True
            else:
                if detected_last:
                    print(f"[{step:4d}] Block lost — using last known position")
                detected_last = False

            # ── Sync real robot → IK model ────────────────────────────────────
            if robot is not None:
                real_joints = robot.get_joint_positions()
                sim_ctrl    = cal.real_to_sim(real_joints, sim_ranges)
                ik_data.ctrl[:6] = sim_ctrl
                ik_data.qpos[:6] = sim_ctrl
                mujoco.mj_forward(ik_model, ik_data)

            # ── State machine + IK step ───────────────────────────────────────
            sm.update_block(block_sim_pos)
            ee_pos = ik.get_ee_position()

            # Navigation finished → advance the pick maneuver between frames.
            # A slice of the generator runs each loop iteration (≈ one video
            # frame's worth of time), so recording continues during the pick.
            if sm.phase == Phase.PICK:
                if robot is None:
                    print("(no robot — skipping perform_pick)")
                    sm.phase = Phase.DONE
                else:
                    if pick_gen is None:
                        pick_gen = perform_pick_steps(robot)
                    t_slice = time.time()
                    try:
                        while time.time() - t_slice < 1.0 / args.fps:
                            next(pick_gen)
                    except StopIteration:
                        sm.phase = Phase.DONE

            target_pos, gripper_action, done = sm.tick(ee_pos)

            if sm.phase not in (Phase.WAIT, Phase.PICK, Phase.DONE):
                dist = np.linalg.norm(ee_pos - target_pos)
                print(f"[{step:4d}] {sm.phase.name:8s} ee=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}) "
                      f"target=({target_pos[0]:+.3f},{target_pos[1]:+.3f},{target_pos[2]:+.3f}) "
                      f"dist={dist*1000:6.1f}mm")

            # Navigation IK only drives the robot before the pick maneuver;
            # during PICK/DONE the generator (or nothing) owns the motors.
            if sm.phase in (Phase.WAIT, Phase.APPROACH, Phase.DESCEND):
                ctrl = ik.step_toward_target(
                    target_pos,
                    gripper_action=gripper_action,
                    gain=IK_GAIN,
                )
                ik_data.ctrl[:] = ctrl
                mujoco.mj_step(ik_model, ik_data)

                # ── Send joint targets to real robot ──────────────────────────
                if robot is not None:
                    target_ctrl  = ik_data.ctrl[:6].copy()
                    real_targets = cal.sim_to_real(target_ctrl, sim_ranges)
                    robot.set_joint_positions(real_targets)

            # ── Sync cube in viz env ──────────────────────────────────────────
            if block_sim_pos is not None:
                viz_env.data.qpos[viz_cube_qpos_adr:viz_cube_qpos_adr + 3] = block_sim_pos
                viz_env.data.qpos[viz_cube_qpos_adr + 3:viz_cube_qpos_adr + 7] = [1, 0, 0, 0]
                viz_env.data.qvel[viz_cube_qpos_adr:viz_cube_qpos_adr + 6] = 0

            # Mirror arm state from IK model → viz env (6 arm joints only —
            # qpos[6] onward is the cube freejoint, which viz owns)
            viz_env.data.ctrl[:] = ik_data.ctrl[:]
            viz_env.data.qpos[:6] = ik_data.qpos[:6]
            viz_env.data.qpos[4] += VIZ_WRIST_ROLL_OFFSET   # render-only roll fix
            viz_env.data.qvel[:6] = ik_data.qvel[:6]
            mujoco.mj_forward(viz_env.model, viz_env.data)

            # ── Panel 1: annotated camera frame ───────────────────────────────
            disp = frame.copy()
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    pts = det_corners[i].reshape(4, 2).astype(np.int32)
                    is_border = int(tid) in tag_centres_mm
                    is_block  = int(tid) == args.block_tag
                    color = (0, 255, 0) if is_border else (0, 200, 255) if is_block else (80, 80, 255)
                    cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, color, 2)
                    cx_px2, cy_px2 = pts.mean(axis=0).astype(int)
                    cv2.putText(disp, str(tid), (cx_px2 - 8, cy_px2 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(disp, f"Camera  phase={sm.phase.name}", (8, disp.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 2: simulation side view ─────────────────────────────────
            sim_rgb = viz_env.render(camera="wide")
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(sim_bgr, f"Sim  ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})",
                        (8, sim_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 3: rectified top-down board view ────────────────────────
            rect = draw_rectified(frame, H, SQUARE_MM, TAG_MM, corners_dict,
                                  tag_centres_mm, args.block_tag, args.block_height,
                                  scale, inset_mm)
            cv2.putText(rect, "Top-down (homography)", (8, rect.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 4: top-down sim camera ──────────────────────────────────
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[0] = 1
            opt.geomgroup[1] = 1
            viz_env._renderer.update_scene(viz_env.data, camera="topdown", scene_option=opt)
            top_rgb = viz_env._renderer.render()
            top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
            top_bgr = cv2.rotate(top_bgr, cv2.ROTATE_90_CLOCKWISE)
            cv2.putText(top_bgr, "Simulation (top-down)", (8, top_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Composite 2×2 frame ───────────────────────────────────────────
            PANEL_W = 480
            PANEL_H = 360

            def fit_panel(img):
                h, w = img.shape[:2]
                scale_f = min(PANEL_W / w, PANEL_H / h)
                nw, nh  = int(w * scale_f), int(h * scale_f)
                resized = cv2.resize(img, (nw, nh))
                out = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                y0  = (PANEL_H - nh) // 2
                x0  = (PANEL_W - nw) // 2
                out[y0:y0+nh, x0:x0+nw] = resized
                return out

            sep_v   = np.full((PANEL_H, 4, 3), 60, dtype=np.uint8)
            top_row = np.hstack([fit_panel(disp), sep_v, fit_panel(sim_bgr)])
            bot_row = np.hstack([fit_panel(rect), sep_v, fit_panel(top_bgr)])
            sep_h   = np.full((4, top_row.shape[1], 3), 60, dtype=np.uint8)
            panels  = np.vstack([top_row, sep_h, bot_row])

            bar_h = 32
            bar   = np.zeros((bar_h, panels.shape[1], 3), dtype=np.uint8)
            if block_sim_pos:
                debug = (f"step={step}  phase={sm.phase.name}  "
                         f"int=({cx_int:.1f},{cy_int:.1f})mm  "
                         f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})m  "
                         f"inliers={n_inliers}")
            else:
                debug = f"step={step}  phase={sm.phase.name}  ID {args.block_tag} not detected  inliers={n_inliers}"
            cv2.putText(bar, debug, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
            combined = np.vstack([bar, panels])

            if writer is None:
                out_h, out_w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
                print(f"Writer ready: {out_w}×{out_h}")

                # Individual raw camera streams (for thesis accuracy analysis)
                run_stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + RUN_SUFFIX
                Path("recordings/camera1").mkdir(parents=True, exist_ok=True)
                cam_h, cam_w = frame.shape[:2]
                cam_writer = cv2.VideoWriter(
                    f"recordings/camera1/{run_stamp}.mp4", fourcc, args.fps, (cam_w, cam_h))
                print(f"Raw stream → recordings/camera1/{run_stamp}.mp4")
                if frame2 is not None:
                    Path("recordings/camera2").mkdir(parents=True, exist_ok=True)
                    cam2_h, cam2_w = frame2.shape[:2]
                    cam2_writer = cv2.VideoWriter(
                        f"recordings/camera2/{run_stamp}.mp4", fourcc, args.fps, (cam2_w, cam2_h))
                    print(f"Raw stream → recordings/camera2/{run_stamp}.mp4")

            writer.write(combined)
            cam_writer.write(frame)               # raw camera 1 frame
            if cam2_writer is not None and frame2 is not None:
                cam2_writer.write(frame2)         # raw camera 2 frame
            frame_count += 1

            # After DONE, keep looping (and recording) for the hold period
            if done:
                if done_time is None:
                    done_time = time.time()
                    print(f"Pick complete at step {step}")
                    print(f"Holding position for {HOLD_AFTER_DONE_S:.0f}s "
                          f"(recording continues, Ctrl-C to end early)...")
                elif time.time() - done_time > HOLD_AFTER_DONE_S:
                    break

            elapsed = time.time() - t_start
            sleep_t = max(0.0, 1.0 / args.fps - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\nStopped at step {frame_count}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved {args.output}")
        if cam_writer is not None:
            cam_writer.release()
        if cam2_writer is not None:
            cam2_writer.release()
        if cap2 is not None:
            cap2.release()
        if robot is not None:
            robot.disable_torque()
            robot.disconnect()
        viz_env.close()


if __name__ == "__main__":
    main()
