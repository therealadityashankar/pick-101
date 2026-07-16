"""pick_tuner.py — Interactive REPL for hand-tuning the pick maneuver.

Run with python -i so you land in an interpreter with the robot connected:

    uv run python -i pick_tuner.py

The robot starts at the pose saved by run_real_ik.py after IK navigation
(calibration/pick_start_positions.json), so you're tuning from the exact
state perform_pick() will start from.

REPL helpers:
    p()                      print current joint positions
    move("wrist_roll", -10)  move a joint RELATIVE by n normalized units
    set_j("gripper", 100)    move a joint to an ABSOLUTE value
    ee()                     print end-effector position (sim coords, metres)
    move_ee(dz=-0.02)        move the gripper tip in Cartesian space via IK
    start()                  return to the saved post-navigation pose
    free()                   torque off — move the arm by hand
    hold()                   torque on — arm holds position
    log                      list of every move you made (paste into perform_pick)

Joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
Values are lerobot-normalized: arm −100..100, gripper 0 (closed) .. 100 (open).
"""

import json
import time
from pathlib import Path

from src.robot.real_robot import RealRobot

PORT       = "/dev/tty.usbmodem5A680089441"
START_POSE = Path("calibration/pick_start_positions.json")

robot = RealRobot(port=PORT)
robot.connect()
robot.enable_torque()

log: list[str] = []   # records your moves as ready-to-paste code lines


def _ramp(joint, target, steps=40, delay=0.02):
    current = robot.get_joint_positions()
    start_val = current[joint]
    for i in range(1, steps + 1):
        t = i / steps
        current[joint] = start_val + t * (target - start_val)
        robot.set_joint_positions(current)
        time.sleep(delay)


def p():
    """Print current joint positions."""
    pos = robot.get_joint_positions()
    for j, v in pos.items():
        print(f"  {j:14s} {v:+8.2f}")
    return pos


def move(joint, delta):
    """Move a joint RELATIVE by delta normalized units."""
    target = robot.get_joint_positions()[joint] + delta
    _ramp(joint, target)
    log.append(f'ramp_joint(robot, "{joint}", pos["{joint}"] + {delta})')
    print(f"{joint} → {target:+.2f}")


def set_j(joint, value):
    """Move a joint to an ABSOLUTE normalized value."""
    _ramp(joint, value)
    log.append(f'ramp_joint(robot, "{joint}", {value})')
    print(f"{joint} → {value:+.2f}")


def start():
    """Return all joints to the saved post-navigation pose."""
    if not START_POSE.exists():
        print(f"No saved pose ({START_POSE}) — run run_real_ik.py first")
        return
    saved = json.loads(START_POSE.read_text())
    current = robot.get_joint_positions()
    steps = 50
    for i in range(1, steps + 1):
        t = i / steps
        interp = {j: current[j] + t * (saved[j] - current[j]) for j in saved}
        robot.set_joint_positions(interp)
        time.sleep(0.02)
    log.clear()
    print("At start pose (log cleared)")


def free():
    """Torque off — move the arm by hand, then p() to read the pose."""
    robot.disable_torque()
    print("Torque OFF — arm is free. hold() to re-engage.")


def hold():
    """Torque on — arm holds position."""
    robot.enable_torque()
    print("Torque ON")


def ramp_joint(_robot, joint, target, steps=40, delay=0.02):
    """Same signature as run_real_ik.ramp_joint — perform_pick lines paste as-is."""
    _ramp(joint, target, steps=steps, delay=delay)
    log.append(f'ramp_joint(robot, "{joint}", {target})')


def ee():
    """Print current end-effector position (sim coords, metres)."""
    import run_real_ik as R
    pos = R.get_ee_pos(robot)
    print(f"ee = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  "
          f"[+x away, +y left, +z up]")
    return pos


def move_ee(dx=0.0, dy=0.0, dz=0.0):
    """Move the gripper tip by (dx, dy, dz) METRES via IK. e.g. move_ee(dz=-0.02)"""
    import run_real_ik as R
    final = R.move_ee(robot, dx, dy, dz)
    log.append(f"move_ee(robot, {dx}, {dy}, {dz})")
    return final


def move_ee_abs(x, y, z):
    """Move the gripper tip to an ABSOLUTE sim position (metres)."""
    import run_real_ik as R
    final = R.move_ee_abs(robot, x, y, z)
    log.append(f"move_ee_abs(robot, {x}, {y}, {z})")
    return final


def show_log():
    """Print your recorded moves as paste-ready perform_pick code."""
    print("\n# ── paste into perform_pick() ──")
    print('pos = robot.get_joint_positions()')
    for line in log:
        print(line)
    print("# ──────────────────────────────\n")


print(__doc__)
if START_POSE.exists():
    print(f"Saved start pose found. Call start() to move there.")
else:
    print("No saved start pose yet — run run_real_ik.py once to create it.")
p()
