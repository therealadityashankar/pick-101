"""Real SO-101 robot interface using lerobot.

Mirrors the sim's action space: receives 4D actions [dx, dy, dz, gripper]
and converts them to joint commands via forward/inverse kinematics.

Joint value ranges (lerobot normalized):
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll: -100 to 100
  gripper: 0 (closed) to 100 (open)

Usage:
    robot = RealRobot(port="/dev/tty.usbmodem585A0076891")
    robot.connect()
    obs = robot.get_joint_positions()  # dict of normalized joint values
    robot.set_joint_positions({"shoulder_pan": 0.0, ...})
    robot.disconnect()
"""

from __future__ import annotations

import time
import numpy as np


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Normalized value ranges per joint
JOINT_RANGES = {
    "shoulder_pan":  (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex":    (-100.0, 100.0),
    "wrist_flex":    (-100.0, 100.0),
    "wrist_roll":    (-100.0, 100.0),
    "gripper":       (0.0, 100.0),
}


class RealRobot:
    """Thin wrapper around lerobot's SO101Follower for sim-compatible control."""

    def __init__(self, port: str, robot_id: str = "so101", use_degrees: bool = False):
        self.port = port
        self.robot_id = robot_id
        self.use_degrees = use_degrees
        self._arm = None

    def connect(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        config = SO101FollowerConfig(port=self.port, id=self.robot_id, use_degrees=self.use_degrees)
        self._arm = SO101Follower(config)
        self._arm.connect()
        self._arm.bus.disable_torque()
        print(f"Connected to SO-101 on {self.port}")

    def disconnect(self):
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None
            print("Disconnected from SO-101")

    def get_joint_positions(self) -> dict[str, float]:
        """Read current normalized joint positions."""
        return self._arm.bus.sync_read("Present_Position", normalize=True)

    def set_joint_positions(self, positions: dict[str, float], normalize: bool = True):
        """Send goal joint positions (normalized by default)."""
        self._arm.bus.sync_write("Goal_Position", positions, normalize=normalize)

    def get_joint_positions_array(self) -> np.ndarray:
        """Return joint positions as array in JOINT_NAMES order."""
        pos = self.get_joint_positions()
        return np.array([pos[j] for j in JOINT_NAMES])

    def set_joint_positions_array(self, positions: np.ndarray):
        """Set joint positions from array in JOINT_NAMES order."""
        self.set_joint_positions({j: float(positions[i]) for i, j in enumerate(JOINT_NAMES)})

    def move_to_home(self, steps: int = 50, delay: float = 0.02):
        """Smoothly move to home position (all zeros, gripper open)."""
        home = {j: 0.0 for j in JOINT_NAMES}
        home["gripper"] = 50.0  # half open
        current = self.get_joint_positions()
        for i in range(1, steps + 1):
            t = i / steps
            interp = {j: current[j] + t * (home[j] - current[j]) for j in JOINT_NAMES}
            self.set_joint_positions(interp)
            time.sleep(delay)
        print("Moved to home position")

    def enable_torque(self):
        """Enable torque on all motors (arm becomes stiff)."""
        self._arm.bus.enable_torque()

    def disable_torque(self):
        """Disable torque on all motors (arm becomes free to move)."""
        self._arm.bus.disable_torque()

    def calibrate(self):
        """Run lerobot's interactive calibration routine."""
        self._arm.calibrate()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
