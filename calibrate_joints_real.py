"""calibrate_joints_real.py — Interactive joint-by-joint calibration for SO-101.

Workflow:
  1. All reference images are rendered in sim BEFORE connecting to the robot.
  2. Calibration: joint by joint, show the pre-rendered image, user moves the
     real arm to match and presses SPACE to capture the reading.
  3. Verification: robot is commanded to specific test poses; pre-rendered
     images are shown so the user can confirm the mapping is correct.

Controls (cv2 window):
    SPACE / Enter  — capture / confirm
    S              — skip this point
    Q              — quit

Usage:
    uv run python calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from src.envs.lift_cube import LiftCubeCartesianEnv

# ── Reference positions per joint ────────────────────────────────────────────
# sim_rad: the angle we'll display in sim and ask the user to match.
# Other joints will be at neutral during rendering — only one joint matters.

CALIBRATION_PLAN = [
    {
        "joint": "shoulder_pan",
        "index": 0,
        "description": "Base rotation — sweeps arm left / right",
        "points": [
            {"label": "Centered (straight forward)", "sim_rad":  0.0},
            {"label": "Swept LEFT  ~90°",             "sim_rad":  1.5708},
            {"label": "Swept RIGHT ~90°",             "sim_rad": -1.5708},
        ],
    },
    {
        "joint": "shoulder_lift",
        "index": 1,
        "description": "Shoulder tilt — raises / lowers the upper arm",
        "points": [
            {"label": "Arm straight UP (vertical)",      "sim_rad":  0.0},
            {"label": "Arm tilted FORWARD ~90°",         "sim_rad":  1.5708},
        ],
    },
    {
        "joint": "elbow_flex",
        "index": 2,
        "description": "Elbow bend — folds the forearm",
        "points": [
            {"label": "Elbow fully STRAIGHT",            "sim_rad":  0.0},
            {"label": "Elbow bent ~90°",                 "sim_rad":  1.5708},
        ],
    },
    {
        "joint": "wrist_flex",
        "index": 3,
        "description": "Wrist pitch — tilts gripper up / down",
        "points": [
            {"label": "Wrist NEUTRAL (level)",           "sim_rad":  0.0},
            {"label": "Wrist flexed DOWN ~90°",          "sim_rad":  1.5708},
        ],
    },
    {
        "joint": "wrist_roll",
        "index": 4,
        "description": "Wrist roll — rotates the gripper",
        "points": [
            {"label": "No roll (level)",                 "sim_rad":  0.0},
            {"label": "Rolled 90° clockwise",            "sim_rad":  1.5708},
            {"label": "Rolled 90° counter-clockwise",    "sim_rad": -1.5708},
        ],
    },
    {
        "joint": "gripper",
        "index": 5,
        "description": "Gripper open / close",
        "points": [
            {"label": "Gripper FULLY OPEN",           "sim_rad": -0.17453},
            {"label": "Gripper FULLY CLOSED",         "sim_rad":  1.74533},
        ],
    },
]

# ── Verification poses (multi-joint, used after calibration) ──────────────────
# ctrl_rad: target angles in sim space for all 6 joints
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

# Verification poses — all chosen so the end effector stays well above the table.
# Keep shoulder_lift small (arm not tilted far forward) and elbow_flex small
# (forearm not folded down). Only pan sweeps test left/right mapping safely.
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
VERIFICATION_POSES = [
    {
        "label": "Home — arm straight up, gripper half-open",
        "ctrl_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    },
    {
        "label": "Swept LEFT — arm still raised",
        "ctrl_rad": [0.8, 0.2, 0.2, 0.0, 0.0, 0.5],
    },
    {
        "label": "Swept RIGHT — arm still raised",
        "ctrl_rad": [-0.8, 0.2, 0.2, 0.0, 0.0, 0.5],
    },
]

# ── Rendering config ──────────────────────────────────────────────────────────
IMG_W = 640
IMG_H = 480
CAMERA = "overview_cam"

JOINT_NAMES  = [j["joint"] for j in CALIBRATION_PLAN]
REAL_RANGES  = [(-100., 100.)] * 5 + [(0., 100.)]


# ─────────────────────────────────────────────────────────────────────────────
# Sim helpers
# ─────────────────────────────────────────────────────────────────────────────

def neutral_ctrl(model) -> np.ndarray:
    ctrl = np.zeros(model.nu)
    ctrl[5] = (model.actuator_ctrlrange[5, 0] + model.actuator_ctrlrange[5, 1]) / 2.0
    return ctrl


def apply_ctrl(data, model, ctrl):
    data.ctrl[:] = ctrl
    data.qpos[:6] = ctrl[:6]
    mujoco.mj_forward(model, data)


def render_frame(renderer, data) -> np.ndarray:
    renderer.update_scene(data, camera=CAMERA)
    rgb = renderer.render()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def annotate(img: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    out = img.copy()
    y = 24
    for text, color in lines:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pre-render all reference images (no robot needed yet)
# ─────────────────────────────────────────────────────────────────────────────

def prerender_all(env, renderer, joints=None):
    """Return nested dict: images[joint][point_index] = annotated BGR image."""
    model = env.model
    data  = env.data
    base  = neutral_ctrl(model)

    if joints is None:
        joints = CALIBRATION_PLAN

    images = {}
    print("Pre-rendering reference images...")

    for jdef in joints:
        jname  = jdef["joint"]
        jidx   = jdef["index"]
        images[jname] = []

        for pt in jdef["points"]:
            target_rad = float(np.clip(
                pt["sim_rad"],
                model.actuator_ctrlrange[jidx, 0],
                model.actuator_ctrlrange[jidx, 1],
            ))
            ctrl = base.copy()
            ctrl[jidx] = target_rad
            apply_ctrl(data, model, ctrl)
            frame = render_frame(renderer, data)

            frame = annotate(frame, [
                (f"Joint: {jname}", (0, 220, 255)),
                (f"Target: {pt['label']}", (255, 255, 255)),
                (f"{np.degrees(target_rad):.1f} deg  ({target_rad:.4f} rad)", (180, 180, 180)),
            ])
            # Reminder that only this joint matters
            msg = "Only THIS joint matters — others can be anywhere"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(frame, msg, (IMG_W // 2 - tw // 2, IMG_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

            images[jname].append(frame)
            print(f"  {jname}  →  {pt['label']}")

    return images


def prerender_verification(env, renderer):
    """Return list of annotated BGR images for verification poses."""
    model = env.model
    data  = env.data
    vimgs = []
    print("Pre-rendering verification poses...")

    for i, pose in enumerate(VERIFICATION_POSES):
        ctrl = np.array(pose["ctrl_rad"], dtype=np.float64)
        # Clip to ctrlrange
        for j in range(len(ctrl)):
            lo, hi = model.actuator_ctrlrange[j]
            ctrl[j] = float(np.clip(ctrl[j], lo, hi))
        apply_ctrl(data, model, ctrl)
        frame = render_frame(renderer, data)
        frame = annotate(frame, [
            (f"Verification pose {i + 1}", (0, 255, 180)),
            (pose["label"], (255, 255, 255)),
            ("Robot will move here — confirm it matches this image", (180, 180, 180)),
        ])
        vimgs.append((frame, ctrl))
        print(f"  Pose {i + 1}: {pose['label']}")

    # Reset to neutral before we hand off to robot phase
    apply_ctrl(data, model, neutral_ctrl(model))
    return vimgs


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear(real_vals, sim_vals):
    if len(real_vals) < 2:
        return {"scale": None, "offset": None, "r2": None}
    x = np.array(real_vals, dtype=np.float64)
    y = np.array(sim_vals, dtype=np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, offset = coef
    y_pred = scale * x + offset
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return {"scale": float(scale), "offset": float(offset), "r2": float(r2)}


def show_capture_screen(window, ref_img, joint, jdesc, pt_label, real_val, captured_count):
    """Display the reference image with live real value and instructions."""
    frame = ref_img.copy()
    h, w  = frame.shape[:2]

    # Live readout bar at bottom-right
    cv2.rectangle(frame, (w - 260, h - 60), (w - 1, h - 1), (30, 30, 30), -1)
    cv2.putText(frame, f"real = {real_val:+.1f}", (w - 250, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, f"captured: {captured_count}", (w - 250, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Instruction bar
    bar = np.zeros((50, w, 3), dtype=np.uint8)
    cv2.putText(bar, f"Match: {pt_label}   |   SPACE = capture    S = skip    Q = quit",
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow(window, np.vstack([frame, bar]))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   required=True)
    parser.add_argument("--output", default="calibration/joint_calibration.json")
    parser.add_argument("--joint",  default=None,
                        help="Calibrate only this joint (e.g. --joint wrist_roll). "
                             "Other joints are loaded from the existing output file.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter plan to single joint if requested
    plan = CALIBRATION_PLAN
    if args.joint is not None:
        names = [j["joint"] for j in CALIBRATION_PLAN]
        if args.joint not in names:
            print(f"Unknown joint '{args.joint}'. Valid names: {names}")
            return
        plan = [j for j in CALIBRATION_PLAN if j["joint"] == args.joint]
        print(f"Single-joint mode: calibrating only '{args.joint}'")

    # ── Phase 1: render everything offline ───────────────────────────────────
    print("\nLoading sim (no robot yet)...")
    env = LiftCubeCartesianEnv(render_mode="rgb_array")
    env.reset()
    cj_id  = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cj_adr = env.model.jnt_qposadr[cj_id]
    env.data.qpos[cj_adr:cj_adr + 3] = [0, 0, -1]
    mujoco.mj_forward(env.model, env.data)

    renderer = mujoco.Renderer(env.model, height=IMG_H, width=IMG_W)

    ref_images  = prerender_all(env, renderer, joints=plan)
    verif_imgs  = prerender_verification(env, renderer)

    sim_ranges  = env.model.actuator_ctrlrange[:6].copy()

    renderer.close()
    env.close()
    print("Sim rendering done.\n")

    # ── Phase 2: connect robot ────────────────────────────────────────────────
    print(f"Connecting to robot on {args.port}...")
    from src.robot.real_robot import RealRobot
    robot = RealRobot(port=args.port)
    robot.connect()
    # Torque OFF so user can move the arm freely
    print("Connected.  Torque is OFF — move the arm freely by hand.\n")

    window = "Joint Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, IMG_W, IMG_H + 50)

    # Load existing calibration so single-joint mode doesn't wipe other joints
    calibration = {}
    if out_path.exists():
        with open(out_path) as f:
            import json as _json
            calibration = _json.load(f)

    try:
        # ── Phase 3: calibrate joint by joint ────────────────────────────────
        for jdef in plan:
            jname  = jdef["joint"]
            jidx   = jdef["index"]
            points = jdef["points"]

            print(f"\n{'='*60}")
            print(f"JOINT {jidx + 1}/6 — {jname.upper()}")
            print(f"  {jdef['description']}")
            print(f"  We will record {len(points)} reference position(s).")
            print(f"  Other joints can be at ANY position — only {jname} matters.")
            print("  Press SPACE in the window to start this joint...")
            while True:
                # Show a "ready" screen so the window stays responsive
                ready_img = np.zeros((IMG_H + 50, IMG_W, 3), dtype=np.uint8)
                cv2.putText(ready_img, f"JOINT {jidx + 1}/6 — {jname.upper()}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                cv2.putText(ready_img, jdef["description"],
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(ready_img, f"Will record {len(points)} reference position(s).",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                cv2.putText(ready_img, f"Only {jname} matters — other joints can be anywhere.",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                cv2.putText(ready_img, "Press SPACE to begin   Q to quit",
                            (10, IMG_H + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
                cv2.imshow(window, ready_img)
                key = cv2.waitKey(100) & 0xFF
                if key in (ord(' '), 13):
                    break
                elif key == ord('q'):
                    raise KeyboardInterrupt
                time.sleep(0.01)

            real_vals = []
            sim_vals  = []

            for pt_idx, pt in enumerate(points):
                target_rad = pt["sim_rad"]
                ref_img    = ref_images[jname][pt_idx]

                print(f"\n  [{pt_idx + 1}/{len(points)}] {pt['label']}")
                print(f"  Move the {jname} to match the image, then press SPACE.")

                done = False
                while not done:
                    try:
                        rjoints  = robot.get_joint_positions()
                        real_val = float(rjoints.get(jname, 0.0))
                    except Exception:
                        real_val = 0.0

                    show_capture_screen(window, ref_img, jname,
                                        jdef["description"], pt["label"],
                                        real_val, len(real_vals))
                    key = cv2.waitKey(50) & 0xFF

                    if key in (ord(' '), 13):
                        real_vals.append(real_val)
                        sim_vals.append(target_rad)
                        print(f"  ✓ Captured: real={real_val:+.2f}  "
                              f"sim_target={target_rad:+.4f}rad")
                        done = True
                    elif key == ord('s'):
                        print("  — Skipped.")
                        done = True
                    elif key == ord('q'):
                        raise KeyboardInterrupt
                    time.sleep(0.01)

            fit = fit_linear(real_vals, sim_vals)
            if fit["scale"] is not None:
                print(f"\n  Fit: sim = {fit['scale']:.5f} × real + {fit['offset']:.5f}"
                      f"   R²={fit['r2']:.4f}")
            else:
                print("\n  Not enough points — joint will use default mapping.")

            calibration[jname] = {
                "index":  jidx,
                "points": [{"real": r, "sim_rad": s}
                           for r, s in zip(real_vals, sim_vals)],
                **fit,
            }

        # ── Phase 4: save calibration ─────────────────────────────────────────
        with open(out_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\nCalibration saved → {out_path}")

        # ── Phase 5: verification ─────────────────────────────────────────────
        print("\n" + "="*60)
        print("VERIFICATION")
        print("The robot will move to specific positions.")
        print("Check if it matches the image shown.")
        print("="*60)

        print("\nEnable torque and run verification poses?")
        confirm_img = np.zeros((IMG_H + 50, IMG_W, 3), dtype=np.uint8)
        cv2.putText(confirm_img, "Run verification?", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        cv2.putText(confirm_img, "Robot will move to test poses. Torque will be enabled.",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(confirm_img, "Press SPACE to proceed   S to skip",
                    (10, IMG_H + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow(window, confirm_img)
        do_verify = False
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord(' '), 13):
                do_verify = True
                break
            elif key == ord('s'):
                break
            time.sleep(0.01)
        if do_verify:
            robot.enable_torque()
            time.sleep(0.3)

            for i, (vimg, ctrl_rad) in enumerate(verif_imgs):
                pose = VERIFICATION_POSES[i]
                print(f"\nPose {i + 1}: {pose['label']}")

                # Convert sim ctrl_rad → real using calibration
                real_targets = {}
                for ji, jname in enumerate(JOINT_NAMES):
                    sim_rad = ctrl_rad[ji]
                    cal = calibration.get(jname, {})
                    if cal.get("scale") is not None and abs(cal["scale"]) > 1e-9:
                        real_val = (sim_rad - cal["offset"]) / cal["scale"]
                    else:
                        lo, hi     = sim_ranges[ji]
                        r_lo, r_hi = REAL_RANGES[ji]
                        t          = (sim_rad - lo) / (hi - lo)
                        real_val   = r_lo + t * (r_hi - r_lo)
                    r_lo, r_hi = REAL_RANGES[ji]
                    real_targets[jname] = float(np.clip(real_val, r_lo, r_hi))

                # ── Step 1: show target image, wait for confirmation ──────────
                preview_bar = np.zeros((50, IMG_W, 3), dtype=np.uint8)
                cv2.putText(preview_bar,
                            f"Pose {i+1}: {pose['label']}  —  "
                            "SPACE = move robot here   S = skip   Q = quit",
                            (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
                cv2.imshow(window, np.vstack([vimg, preview_bar]))

                confirmed = False
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key in (ord(' '), 13):
                        confirmed = True
                        break
                    elif key == ord('s'):
                        print("  — Skipped.")
                        break
                    elif key == ord('q'):
                        raise KeyboardInterrupt
                    time.sleep(0.01)

                if not confirmed:
                    continue

                # ── Step 2: move slowly via interpolation ─────────────────────
                print(f"  Moving to: {', '.join(f'{k}={v:+.1f}' for k, v in real_targets.items())}")
                current = robot.get_joint_positions()
                steps, delay = 60, 0.03   # ~1.8 s total, slow and smooth
                for s in range(1, steps + 1):
                    t = s / steps
                    interp = {j: current[j] + t * (real_targets[j] - current[j])
                              for j in real_targets}
                    robot.set_joint_positions(interp)
                    time.sleep(delay)
                time.sleep(0.4)   # settle

                # ── Step 3: show image again, ask if it matches ───────────────
                check_bar = np.zeros((50, IMG_W, 3), dtype=np.uint8)
                cv2.putText(check_bar,
                            "Does the robot match the image?   Y = yes   N = no   Q = quit",
                            (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
                cv2.imshow(window, np.vstack([vimg, check_bar]))

                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('y'):
                        print("  ✓ Looks good!")
                        break
                    elif key == ord('n'):
                        print("  ✗ Mismatch noted — recalibrate this joint.")
                        break
                    elif key == ord('q'):
                        raise KeyboardInterrupt
                    time.sleep(0.01)

            robot.disable_torque()

    except KeyboardInterrupt:
        print("\nInterrupted — saving what we have.")
        with open(out_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"Saved → {out_path}")
    finally:
        cv2.destroyAllWindows()
        try:
            robot.disable_torque()
            robot.disconnect()
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
