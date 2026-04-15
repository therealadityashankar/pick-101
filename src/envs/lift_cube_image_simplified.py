"""Lift cube environment with vision-based observation space.

Like lift_cube.py but replaces privileged state with segmentation-derived 2D
image coordinates observed from the wrist camera:
  - static fingertip topmost point (x, y) in [0,1]
  - moving fingertip topmost point (x, y) in [0,1]
  - cube top-face convex corners: 4 points × (x, y) in [0,1],
    sorted clockwise from topmost corner
  - gripper opening scalar

Obs dim: 13
Action dim: 4 (delta XYZ + gripper)

Supports camera pose noise augmentation at reset for sim-to-real robustness.
"""
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.controllers.ik_controller import IKController
from src.envs.rewards import REWARD_FUNCTIONS


class LiftCubeImageEnv(gym.Env):
    """Lift cube with Cartesian action space and vision-based observations.

    Action space (4 dims):
        - Delta X, Y, Z for end-effector position
        - Gripper open/close (-1 to 1)

    Observation space (9 dims, all in [0, 1]):
        - static_fingertip_x, static_fingertip_y     (projected to wrist cam)
        - moving_fingertip_x, moving_fingertip_y     (projected to wrist cam)
        - cube_bbox_x1, cube_bbox_y1                 (top-left corner)
        - cube_bbox_x2, cube_bbox_y2                 (bottom-right corner)
        - gripper_opening                             (normalised to [0, 1])

    Camera noise:
        At each reset, the wrist_cam extrinsics are perturbed by a small
        random offset (translation + rotation) sampled uniformly within the
        specified bounds.  This forces the policy to be robust to camera
        mounting errors and is the primary domain-randomisation technique
        for sim-to-real transfer.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Wrist cam render resolution (matches real innoMaker camera crop)
    CAM_W = 640
    CAM_H = 480

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.02,
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "dense",
        reward_version: str = "v11",
        curriculum_stage: int = 0,
        lock_wrist: bool = False,
        # Camera noise augmentation (set to 0 to disable)
        cam_pos_noise: float = 0.003,    # ±3 mm translation noise per axis
        cam_euler_noise: float = 0.05,   # ±~3° rotation noise per axis (radians)
        # Observation noise augmentation (set to 0 to disable)
        obs_noise: float = 0.01,         # ±1% of normalised image space per keypoint
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.reward_type = reward_type
        self.reward_version = reward_version
        self.curriculum_stage = curriculum_stage
        self.lock_wrist = lock_wrist
        self.cam_pos_noise = cam_pos_noise
        self.cam_euler_noise = cam_euler_noise
        self.obs_noise = obs_noise

        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._reset_gripper_action = None
        self._prev_action = np.zeros(4)
        self._open_gripper_count = 0

        # Load model
        scene_path = Path(__file__).parent.parent.parent / "models/so101/lift_cube.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Geom IDs for contact detection (physics pads)
        self._static_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self._moving_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._cube_top_face_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face")
        # Geom IDs for segmentation-based obs (markers sit above cube, always visible)
        self._static_marker_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_fingertip_marker")
        self._moving_marker_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_fingertip_marker")

        # Wrist camera ID
        self._wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        # Store nominal wrist cam pos/euler from XML for noise augmentation
        self._wrist_cam_nominal_pos = self.model.cam_pos[self._wrist_cam_id].copy()
        self._wrist_cam_nominal_euler = self._quat_to_euler(self.model.cam_quat[self._wrist_cam_id])

        # IK controller
        self.ik = IKController(self.model, self.data, end_effector_site="gripperframe")

        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 13-dim, all values in [0, 1]
        # [static_x, static_y, moving_x, moving_y, c0x,c0y, c1x,c1y, c2x,c2y, c3x,c3y, gripper]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        # Target EE position tracker
        self._target_ee_pos = None

        # Segmentation renderer created lazily on first obs call (avoids Metal init in __init__)
        self._seg_renderer = None
        self._crop_x = (self.CAM_W - self.CAM_H) // 2  # pixels to crop each side → square

        # Eval renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    # ------------------------------------------------------------------
    # Segmentation-based observation helpers
    # ------------------------------------------------------------------

    def _render_seg(self, hide_cube: bool = False) -> np.ndarray:
        """Render segmentation from wrist cam, return cropped (480, 480) geom ID map."""
        if self._seg_renderer is None:
            self._seg_renderer = mujoco.Renderer(self.model, height=self.CAM_H, width=self.CAM_W)

        if hide_cube:
            orig_cube_group = self.model.geom_group[self._cube_geom_id]
            orig_top_group = self.model.geom_group[self._cube_top_face_geom_id]
            self.model.geom_group[self._cube_geom_id] = 4
            self.model.geom_group[self._cube_top_face_geom_id] = 4
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[4] = 0
            self._seg_renderer.update_scene(self.data, camera="wrist_cam", scene_option=scene_option)
        else:
            self._seg_renderer.update_scene(self.data, camera="wrist_cam")

        self._seg_renderer.enable_segmentation_rendering()
        seg = self._seg_renderer.render()
        self._seg_renderer.disable_segmentation_rendering()

        if hide_cube:
            self.model.geom_group[self._cube_geom_id] = orig_cube_group
            self.model.geom_group[self._cube_top_face_geom_id] = orig_top_group

        return seg[..., 0][:, self._crop_x: self._crop_x + self.CAM_H]  # (480, 480)

    def _topmost_point_norm(self, id_map: np.ndarray, geom_id: int) -> tuple[float, float]:
        """Return normalised (x, y) of topmost pixel for geom_id, or (0, 0) if not visible."""
        ys, xs = np.where(id_map == geom_id)
        if len(ys) == 0:
            return 0.0, 0.0
        idx = np.argmin(ys)
        return float(xs[idx]) / self.CAM_H, float(ys[idx]) / self.CAM_H

    def _cube_corners_norm(self, id_map: np.ndarray) -> np.ndarray:
        """Return (4, 2) normalised corners of cube top face, clockwise from topmost.

        Falls back to zeros if the geom is not visible or has too few pixels.
        """
        ys, xs = np.where(id_map == self._cube_top_face_geom_id)
        if len(ys) < 4:
            return np.zeros((4, 2), dtype=np.float32)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        hull = cv2.convexHull(pts)
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)  # (N, 2)

        # Pad or trim to exactly 4 corners
        if len(approx) < 4:
            pad = np.tile(approx.mean(axis=0), (4 - len(approx), 1))
            approx = np.vstack([approx, pad])
        elif len(approx) > 4:
            # Keep the 4 points that form the largest area quadrilateral (drop extras)
            approx = approx[:4]

        # Sort clockwise from topmost corner
        cx, cy = approx.mean(axis=0)
        angles = np.arctan2(approx[:, 1] - cy, approx[:, 0] - cx)
        # Start from topmost (min y = most negative sin component)
        order = np.argsort(angles)
        approx = approx[order]

        # Normalise to [0, 1] using the square crop size (CAM_H = 480)
        return (approx / self.CAM_H).astype(np.float32)

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Two renders: cube hidden for fingertips, normal for cube corners
        id_map_no_cube = self._render_seg(hide_cube=True)
        id_map_with_cube = self._render_seg(hide_cube=False)

        # Fingertip topmost points (normalised) — use cube-hidden map
        su, sv = self._topmost_point_norm(id_map_no_cube, self._static_marker_geom_id)
        mu, mv = self._topmost_point_norm(id_map_no_cube, self._moving_marker_geom_id)

        # Cube top-face convex corners (4 × 2, normalised, clockwise from topmost)
        corners = self._cube_corners_norm(id_map_with_cube).flatten()  # (8,)

        # Gripper opening normalised to [0, 1]
        gripper_raw = self._get_gripper_state()
        lo, hi = self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
        gripper_norm = float(np.clip((gripper_raw - lo) / (hi - lo), 0.0, 1.0))

        obs = np.array([su, sv, mu, mv, *corners, gripper_norm], dtype=np.float32)
        if self.obs_noise > 0.0 and self.np_random is not None:
            noise = self.np_random.uniform(-self.obs_noise, self.obs_noise, size=obs.shape).astype(np.float32)
            noise[-1] = 0.0  # don't noise gripper opening — it's a direct sensor reading
            obs = np.clip(obs + noise, 0.0, 1.0)
        return obs

    # ------------------------------------------------------------------
    # Camera noise augmentation
    # ------------------------------------------------------------------

    def _apply_camera_noise(self):
        """Perturb wrist cam extrinsics by small random offsets."""
        if self.cam_pos_noise == 0 and self.cam_euler_noise == 0:
            return
        if self.np_random is None:
            return

        # Position noise
        pos_delta = self.np_random.uniform(
            -self.cam_pos_noise, self.cam_pos_noise, size=3
        )
        self.model.cam_pos[self._wrist_cam_id] = self._wrist_cam_nominal_pos + pos_delta

        # Rotation noise — perturb euler then convert back to quat
        euler_delta = self.np_random.uniform(
            -self.cam_euler_noise, self.cam_euler_noise, size=3
        )
        noisy_euler = self._wrist_cam_nominal_euler + euler_delta
        self.model.cam_quat[self._wrist_cam_id] = self._euler_to_quat(noisy_euler)

    @staticmethod
    def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
        """Convert euler angles (roll, pitch, yaw) to quaternion (w,x,y,z)."""
        r, p, y = euler / 2.0
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y_ = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return np.array([w, x, y_, z])

    # ------------------------------------------------------------------
    # Shared helpers (unchanged from lift_cube.py)
    # ------------------------------------------------------------------

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return float(self.data.qpos[gripper_qpos_addr])

    def _check_cube_contacts(self) -> tuple[bool, bool]:
        has_static_contact = False
        has_moving_contact = False
        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2
            other_geom = None
            if geom1 == self._cube_geom_id:
                other_geom = geom2
            elif geom2 == self._cube_geom_id:
                other_geom = geom1
            if other_geom is not None:
                if other_geom == self._static_pad_geom_id:
                    has_static_contact = True
                if other_geom == self._moving_pad_geom_id:
                    has_moving_contact = True
        return has_static_contact, has_moving_contact

    def _is_grasping(self) -> bool:
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.25
        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        return is_closed and (has_gripper_contact and has_jaw_contact)

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        cube_pos = self.data.sensor("cube_pos").data.copy()
        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and cube_z > self.lift_height
        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_gripper_contact": has_gripper_contact,
            "has_jaw_contact": has_jaw_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "hold_count": self._hold_count,
            "is_success": self._hold_count >= self.hold_steps,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        if self.curriculum_stage == 0:
            if self.np_random is not None:
                cube_x = 0.40 + self.np_random.uniform(-0.03, 0.03)
                cube_y = -0.10 + self.np_random.uniform(-0.03, 0.03)
                self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, 0.01]
                self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        elif self.curriculum_stage == 1:
            self._reset_with_cube_in_gripper(cube_qpos_addr, lift_height=self.lift_height)
        elif self.curriculum_stage == 2:
            self._reset_with_cube_in_gripper(cube_qpos_addr, lift_height=0.03)
        elif self.curriculum_stage == 3:
            self._reset_gripper_near_cube(cube_qpos_addr)
        elif self.curriculum_stage == 4:
            self._reset_gripper_far_from_cube(cube_qpos_addr)

        # Apply camera noise after physics reset
        self._apply_camera_noise()

        mujoco.mj_forward(self.model, self.data)
        self._target_ee_pos = self.ik.get_ee_position().copy()

        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._prev_action = np.zeros(4)
        self._open_gripper_count = 0

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Curriculum reset helpers (identical to lift_cube.py)
    # ------------------------------------------------------------------

    def _reset_with_cube_in_gripper(self, cube_qpos_addr: int, lift_height: float):
        height_offset = 0.03
        gripper_open = 0.3
        gripper_closed = -0.8
        grasp_z_offset = 0.005
        finger_width_offset = -0.015
        locked_joints = [3, 4]

        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0
        cube_z = 0.015

        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        actual_cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3].copy()
        above_pos = actual_cube_pos.copy()
        above_pos[2] += grasp_z_offset + height_offset
        above_pos[1] += finger_width_offset
        for _ in range(300):
            ctrl = self.ik.step_toward_target(above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        grasp_target = actual_cube_pos.copy()
        grasp_target[2] += grasp_z_offset
        grasp_target[1] += finger_width_offset
        for _ in range(200):
            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        contact_step = None
        contact_action = None
        tighten_amount = 0.4
        grasp_action = gripper_closed

        for step in range(300):
            if contact_step is None:
                t = min(step / 250, 1.0)
                gripper = gripper_open - 2.0 * t
            else:
                steps_since = step - contact_step
                t_slow = min(steps_since / 100, 1.0)
                target_action = max(contact_action - tighten_amount, -1.0)
                gripper = contact_action + (target_action - contact_action) * t_slow

            ctrl = self.ik.step_toward_target(grasp_target, gripper_action=gripper, gain=0.5, locked_joints=locked_joints)
            ctrl[3] = np.pi / 2
            ctrl[4] = np.pi / 2
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

            if self._is_grasping() and contact_step is None:
                contact_step = step
                contact_action = gripper
            if contact_step is not None:
                target_action = max(contact_action - tighten_amount, -1.0)
                if gripper <= target_action + 0.01:
                    grasp_action = gripper
                    break

        if contact_step is not None and gripper > max(contact_action - tighten_amount, -1.0) + 0.01:
            grasp_action = gripper
        self._reset_gripper_action = grasp_action

    def _reset_gripper_near_cube(self, cube_qpos_addr: int):
        height_offset = 0.03
        gripper_open = 0.3
        grasp_z_offset = 0.005
        finger_width_offset = -0.015

        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0

        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        actual_cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3].copy()
        above_pos = actual_cube_pos.copy()
        above_pos[1] += finger_width_offset
        above_pos[2] = actual_cube_pos[2] + grasp_z_offset + height_offset

        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(100):
            ctrl = self.ik.step_toward_target(above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def _reset_gripper_far_from_cube(self, cube_qpos_addr: int):
        gripper_open = 0.3

        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0

        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        actual_cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3].copy()
        if self.np_random is not None:
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(0.08, 0.12)
        else:
            angle, distance = 0.0, 0.10

        start_pos = np.array([
            np.clip(actual_cube_pos[0] + distance * np.cos(angle), 0.15, 0.45),
            np.clip(actual_cube_pos[1] + distance * np.sin(angle), -0.25, 0.25),
            self.lift_height + 0.02,
        ])

        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(150):
            ctrl = self.ik.step_toward_target(start_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        delta_xyz = action[:3] * self.action_scale
        gripper_action = action[3]

        self._target_ee_pos += delta_xyz
        self._target_ee_pos[0] = np.clip(self._target_ee_pos[0], 0.1, 0.5)
        self._target_ee_pos[1] = np.clip(self._target_ee_pos[1], -0.3, 0.3)
        self._target_ee_pos[2] = np.clip(self._target_ee_pos[2], 0.01, 0.4)

        if self.lock_wrist:
            stable_gripper = self._reset_gripper_action if self._reset_gripper_action is not None else gripper_action
            ctrl = self.ik.step_toward_target(self._target_ee_pos, gripper_action=stable_gripper, gain=0.5, locked_joints=[4])
            ctrl[4] = np.pi / 2
        else:
            ctrl = self.ik.step_toward_target(self._target_ee_pos, gripper_action=gripper_action, gain=0.5)

        self.data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        if info["is_lifted"]:
            self._hold_count += 1
        else:
            self._hold_count = 0
        info["hold_count"] = self._hold_count

        is_success = self._hold_count >= self.hold_steps
        info["is_success"] = is_success

        reward = self._compute_reward(info, was_grasping=self._was_grasping, action=action)
        self._was_grasping = info["is_grasping"]
        self._prev_action = action.copy()

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        if self.reward_type == "sparse":
            return 0.0 if info["is_success"] else -1.0
        reward_fn = REWARD_FUNCTIONS.get(self.reward_version)
        if reward_fn is None:
            raise ValueError(f"Unknown reward version: {self.reward_version}")
        return reward_fn(self, info, was_grasping=was_grasping, action=action)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, camera: str = "closeup") -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            cam = mujoco.MjvCamera()
            if camera == "closeup":
                cam.lookat[:] = [0.40, -0.10, 0.03]
                cam.distance = 0.35
                cam.azimuth = 90
                cam.elevation = -15
            elif camera == "wide":
                cam.lookat[:] = [0.25, -0.05, 0.05]
                cam.distance = 0.8
                cam.azimuth = 135
                cam.elevation = -25
            else:
                cam.lookat[:] = [0.25, -0.05, 0.05]
                cam.distance = 0.8
                cam.azimuth = 45
                cam.elevation = -25
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        return None

    def close(self):
        if self._seg_renderer is not None:
            self._seg_renderer.close()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
