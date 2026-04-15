"""Quick connection test for the real SO-101 arm.

Usage:
    python test_real_robot.py --port /dev/tty.usbmodem585A0076891
    python test_real_robot.py --port /dev/tty.usbmodem585A0076891 --calibrate
"""
import argparse
import time
from src.robot.real_robot import RealRobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port for SO-101 (e.g. /dev/tty.usbmodem...)")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration routine")
    args = parser.parse_args()

    with RealRobot(port=args.port) as robot:
        if args.calibrate:
            robot.calibrate()
            return

        print("\nCurrent joint positions:")
        pos = robot.get_joint_positions()
        for name, val in pos.items():
            print(f"  {name:20s}: {val:.2f}")

        print("\nReading positions for 5 seconds (move the arm to verify)...")
        for _ in range(10):
            pos = robot.get_joint_positions()
            vals = "  ".join(f"{n[:4]}={v:.1f}" for n, v in pos.items())
            print(f"  {vals}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
