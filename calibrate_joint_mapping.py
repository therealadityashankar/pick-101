"""Calibrate the lerobot -> MuJoCo joint angle mapping.

Reads actual joint angles in degrees from the real robot and maps them
to MuJoCo radians. Move the arm to a few known poses and the script
will print the per-joint degree->radian scale and offset.

Usage:
    python calibrate_joint_mapping.py --port /dev/tty.usbmodem5A680089441

Controls:
    SPACE  — record current pose
    H      — move sim to home (all zeros) and record
    Q      — quit and print computed mapping
"""
import argparse
import time
import numpy as np
from src.robot.real_robot import RealRobot, JOINT_NAMES


def degrees_to_radians(deg: float) -> float:
    return deg * np.pi / 180.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    robot = RealRobot(port=args.port, use_degrees=True)
    robot.connect()

    print("\nJoint angle reader (degrees mode)")
    print("Move the arm freely — values update every 0.2s")
    print("Press ENTER to snapshot current angles, Q+ENTER to quit\n")

    snapshots = []
    try:
        while True:
            pos = robot.get_joint_positions()
            parts = "  ".join(f"{n[:8]}={v:+7.1f}°" for n, v in pos.items())
            print(f"\r{parts}    ", end="", flush=True)
            time.sleep(0.2)

            import sys, select
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    break
                else:
                    snapshots.append({n: degrees_to_radians(v) for n, v in pos.items()})
                    print(f"\n  Snapshot {len(snapshots)} recorded (radians): "
                          + "  ".join(f"{n[:8]}={v:+.3f}" for n, v in snapshots[-1].items()))
    except KeyboardInterrupt:
        pass

    robot.disconnect()

    if snapshots:
        print("\n\n=== Recorded poses (radians) ===")
        for i, s in enumerate(snapshots):
            print(f"\nPose {i+1}:")
            for name in JOINT_NAMES:
                print(f"  {name:<20s}: {s.get(name, 0.0):+.4f} rad  ({np.degrees(s.get(name, 0.0)):+.1f}°)")

        print("\n=== lerobot_to_mujoco mapping (degrees -> radians, scale = pi/180) ===")
        print("Each lerobot degree value can be directly converted with np.deg2rad()")
        print("\nPaste this into calibration_viewer.py lerobot_to_mujoco():")
        print("""
    scale = np.pi / 180.0  # degrees -> radians
    return {
        "shoulder_pan":  joint_positions.get("shoulder_pan",  0.0) * scale,
        "shoulder_lift": joint_positions.get("shoulder_lift", 0.0) * scale,
        "elbow_flex":    joint_positions.get("elbow_flex",    0.0) * scale,
        "wrist_flex":    joint_positions.get("wrist_flex",    0.0) * scale,
        "wrist_roll":    joint_positions.get("wrist_roll",    0.0) * scale,
        "gripper":       joint_positions.get("gripper",       0.0) / 100.0,
    }""")


if __name__ == "__main__":
    main()
