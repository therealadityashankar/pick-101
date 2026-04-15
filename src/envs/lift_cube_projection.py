"""Lift cube environment with projection-based observation space.

Like lift_cube_image_simplified.py but replaces segmentation rendering with
analytic 3D->2D projection. No MuJoCo renderer needed — fully parallelisable.

Obs dim: 13
Action dim: 4 (delta XYZ + gripper)
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import spaces

from src.controllers.ik_controller import IKController
from src.envs.rewards import REWARD_FUNCTIONS


class LiftCubeProjectionEnv(gym.Env):
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
        # Optional: path to .npy file of saved qpos snapshots from a prior stage
        straddle_states_path: str | None = None,
        # If True, the cube is frozen in place (position reset every step)
        freeze_cube: bool = False,
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
        self.straddle_states_path = straddle_states_path
        self.freeze_cube = freeze_cube

        # Load saved qpos snapshots for straddle-start resets
        self._straddle_states: np.ndarray | None = None
        if straddle_states_path is not None:
            p = Path(straddle_states_path)
            if not p.is_absolute():
                # Resolve relative to repo root (parent of this file's package)
                p = Path(__file__).resolve().parents[2] / p
            self._straddle_states = np.load(p)

        self._step_count = 0
        self._hold_count = 0
        self._was_grasping = False
        self._reset_gripper_action = None
        self._prev_action = np.zeros(4)
        self._open_gripper_count = 0
        self._frozen_cube_qpos = None  # set after reset positions are determined

        # Load model
        scene_path = Path(__file__).parent.parent.parent / "models/so101/lift_cube_simplified.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Geom IDs for contact detection (physics pads)
        self._static_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self._moving_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        # Site IDs for projection-based fingertip obs
        self._static_fingertip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")
        self._moving_fingertip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "moving_fingertip")
        # Cube body ID for projection-based corner obs
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        # Half-height of the cube geom (used for top-face z offset)
        self._cube_half_z = float(self.model.geom_size[self._cube_geom_id][2])

        # Wrist camera ID
        self._wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        # Store nominal wrist cam pos/euler from XML for noise augmentation
        self._wrist_cam_nominal_pos = self.model.cam_pos[self._wrist_cam_id].copy()
        self._wrist_cam_nominal_euler = self._quat_to_euler(self.model.cam_quat[self._wrist_cam_id])
        # Current noise offsets (sampled at each reset, applied in _get_camera_matrices)
        self._cam_pos_offset = np.zeros(3)
        self._cam_euler_offset = np.zeros(3)

        # IK controller
        self.ik = IKController(self.model, self.data, end_effector_site="gripperframe")

        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Cache gripper joint range for normalisation
        _gripper_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self._gripper_lo, self._gripper_hi = self.model.jnt_range[_gripper_jid]

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 13-dim, all values in [0, 1]
        # [static_x, static_y, moving_x, moving_y, c0x,c0y, c1x,c1y, c2x,c2y, c3x,c3y, gripper]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        # Target EE position tracker
        self._target_ee_pos = None

        self._crop_x = (self.CAM_W - self.CAM_H) // 2  # pixels to crop each side → square

        # Eval renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    # ------------------------------------------------------------------
    # Segmentation-based observation helpers
    # ------------------------------------------------------------------

    def _get_camera_matrices(self):
        """Return (K, R, t) for the wrist cam in OpenCV convention."""
        fovy = float(self.model.cam_fovy[self._wrist_cam_id])
        f = (self.CAM_H / 2.0) / jnp.tan(jnp.deg2rad(fovy / 2.0))
        cx, cy = self.CAM_W / 2.0, self.CAM_H / 2.0
        K = jnp.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        R_mj2cv = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=jnp.float32)
        cam_xmat = jnp.array(self.data.cam_xmat[self._wrist_cam_id].reshape(3, 3))
        delta_R = jnp.array(self._euler_to_mat(self._cam_euler_offset))
        noisy_cam_xmat = cam_xmat @ delta_R
        R = R_mj2cv @ noisy_cam_xmat.T
        noisy_cam_pos = jnp.array(self.data.cam_xpos[self._wrist_cam_id]) + jnp.array(self._cam_pos_offset)
        t = -R @ noisy_cam_pos
        return K, R, t

    def _project_points_norm(self, points_3d, K, R, t) -> np.ndarray:
        """Project (N, 3) world points to (N, 2) normalised coords in cropped 480x480 space."""
        pts = jnp.array(points_3d)
        pts_cam = (R @ pts.T).T + t
        pts_proj = (K @ pts_cam.T).T
        px = pts_proj[:, 0] / pts_proj[:, 2]
        py = pts_proj[:, 1] / pts_proj[:, 2]
        px_cropped = (px - self._crop_x) / self.CAM_H
        py_cropped = py / self.CAM_H
        return np.asarray(jnp.stack([px_cropped, py_cropped], axis=1), dtype=np.float32)

    def _get_cube_top_corners_world(self) -> np.ndarray:
        """Return (4, 3) world-space corners of cube top face."""
        cube_pos = jnp.array(self.data.xpos[self._cube_body_id])
        cube_mat = jnp.array(self.data.xmat[self._cube_body_id].reshape(3, 3))
        half_xy = float(self.model.geom_size[self._cube_geom_id][0])  # width
        half_z  = float(self.model.geom_size[self._cube_geom_id][2])  # height
        local_corners = jnp.array([
            [ half_xy,  half_xy, half_z],
            [-half_xy,  half_xy, half_z],
            [-half_xy, -half_xy, half_z],
            [ half_xy, -half_xy, half_z],
        ])
        return np.asarray((cube_mat @ local_corners.T).T + cube_pos)

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
        K, R, t = self._get_camera_matrices()

        # Fingertip positions from site world coords
        static_pos = self.data.site_xpos[self._static_fingertip_site_id].reshape(1, 3)
        moving_pos = self.data.site_xpos[self._moving_fingertip_site_id].reshape(1, 3)
        su, sv = self._project_points_norm(static_pos, K, R, t)[0]
        mu, mv = self._project_points_norm(moving_pos, K, R, t)[0]

        # Cube top-face corners from body position + known offsets
        world_corners = self._get_cube_top_corners_world()
        corners = self._project_points_norm(world_corners, K, R, t).flatten()  # (8,)

        # Gripper opening normalised to [0, 1]
        gripper_raw = self._get_gripper_state()
        gripper_norm = float(np.clip((gripper_raw - self._gripper_lo) / (self._gripper_hi - self._gripper_lo), 0.0, 1.0))

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
        """Sample new camera noise offsets for this episode (does not mutate model)."""
        if self.np_random is None or (self.cam_pos_noise == 0 and self.cam_euler_noise == 0):
            self._cam_pos_offset = np.zeros(3)
            self._cam_euler_offset = np.zeros(3)
            return
        self._cam_pos_offset = self.np_random.uniform(
            -self.cam_pos_noise, self.cam_pos_noise, size=3
        )
        self._cam_euler_offset = self.np_random.uniform(
            -self.cam_euler_noise, self.cam_euler_noise, size=3
        )

    @staticmethod
    def _euler_to_mat(euler: np.ndarray) -> np.ndarray:
        """Convert euler angles (roll, pitch, yaw) to 3x3 rotation matrix."""
        r, p, y = euler
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [  -sp,            cp*sr,            cp*cr],
        ], dtype=np.float64)

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
        ncon = self.data.ncon
        if ncon == 0:
            return False, False
        geom1 = self.data.contact.geom1[:ncon]
        geom2 = self.data.contact.geom2[:ncon]
        cube_mask = (geom1 == self._cube_geom_id) | (geom2 == self._cube_geom_id)
        other = np.where(geom1 == self._cube_geom_id, geom2, geom1)
        has_static_contact = bool(np.any(cube_mask & (other == self._static_pad_geom_id)))
        has_moving_contact = bool(np.any(cube_mask & (other == self._moving_pad_geom_id)))
        return has_static_contact, has_moving_contact

    def _is_grasping(self) -> bool:
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.25
        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        return is_closed and (has_gripper_contact and has_jaw_contact)

    def _cube_upright_frac(self) -> float:
        """Return how upright the cube is: 1.0 = perfectly upright, 0.0 = on its side.

        Uses the dot product of the cube's local Z axis with world Z.
        """
        cube_mat = self.data.xmat[self._cube_body_id].reshape(3, 3)
        world_z = np.array([0.0, 0.0, 1.0])
        return float(np.clip(np.dot(cube_mat[:, 2], world_z), 0.0, 1.0))

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        cube_pos = self.data.sensor("cube_pos").data.copy()
        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and cube_z > self.lift_height
        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        cube_upright = self._cube_upright_frac()

        # Fraction of cube top-face corners visible in wrist cam frame
        K, R, t = self._get_camera_matrices()
        world_corners = self._get_cube_top_corners_world()
        projected = self._project_points_norm(world_corners, K, R, t)  # (4, 2)
        corners_in_frame = float(np.mean((projected >= 0.0).all(axis=1) & (projected <= 1.0).all(axis=1)))

        static_pos = self.data.site_xpos[self._static_fingertip_site_id].copy()
        moving_pos = self.data.site_xpos[self._moving_fingertip_site_id].copy()

        # Success condition per reward version
        if "zone" in self.reward_version:
            K, R, t = self._get_camera_matrices()
            static_proj = self._project_points_norm(
                self.data.site_xpos[self._static_fingertip_site_id].reshape(1, 3), K, R, t)[0]
            moving_proj = self._project_points_norm(
                self.data.site_xpos[self._moving_fingertip_site_id].reshape(1, 3), K, R, t)[0]
            world_corners = self._get_cube_top_corners_world()
            cube_proj_center = self._project_points_norm(world_corners, K, R, t).mean(axis=0)
            grip_mid = (static_proj + moving_proj) / 2.0
            center_dist = float(np.linalg.norm(grip_mid - cube_proj_center))
            grip_axis_2d = moving_proj - static_proj
            grip_axis_2d = grip_axis_2d / (np.linalg.norm(grip_axis_2d) + 1e-8)
            static_side_2d = float(np.dot(static_proj - cube_proj_center, grip_axis_2d))
            moving_side_2d = float(np.dot(moving_proj - cube_proj_center, grip_axis_2d))
            is_success = (
                cube_upright > 0.85
                and center_dist < 0.10
                and static_side_2d < 0.0
                and moving_side_2d > 0.0
            )
        elif "straddle" in self.reward_version:
            cube_top_z = cube_z + self._cube_half_z
            finger_vec = moving_pos - static_pos
            finger_axis = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
            static_side = np.dot(static_pos - cube_pos, finger_axis)
            moving_side = np.dot(moving_pos - cube_pos, finger_axis)
            # Gripper midpoint should be over cube center (zone condition)
            grip_mid = (static_pos + moving_pos) / 2.0
            mid_to_cube_xy = np.linalg.norm(grip_mid[:2] - cube_pos[:2])
            is_success = (
                static_pos[2] < cube_top_z       # both fingertips below cube top
                and moving_pos[2] < cube_top_z
                and static_side < 0.0             # straddling opposite sides
                and moving_side > 0.0
                and mid_to_cube_xy < 0.04         # gripper center over cube (4cm leeway)
            )
        elif "grip" in self.reward_version:
            is_success = is_grasping
        else:
            is_success = self._hold_count >= self.hold_steps

        # Cube linear velocity (from free joint qvel)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qvel_addr = self.model.jnt_dofadr[cube_joint_id]
        cube_vel = float(np.linalg.norm(self.data.qvel[cube_qvel_addr:cube_qvel_addr + 3]))

        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "cube_top_z": cube_z + self._cube_half_z,
            "cube_vel": cube_vel,
            "cube_upright": cube_upright,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_gripper_contact": has_gripper_contact,
            "has_jaw_contact": has_jaw_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "hold_count": self._hold_count,
            "is_success": is_success,
            "corners_in_frame": corners_in_frame,
            "static_fingertip_pos": static_pos,
            "moving_fingertip_pos": moving_pos,
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

        if self._straddle_states is not None:
            self._reset_from_straddle_states()
        elif self.curriculum_stage == 0:
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

        # Snapshot cube position for freeze_cube mode
        if self.freeze_cube:
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
            qa = self.model.jnt_qposadr[cube_joint_id]
            self._frozen_cube_qpos = self.data.qpos[qa:qa + 7].copy()

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
        height_offset = 0.12
        gripper_open = 0.3
        grasp_z_offset = 0.005
        finger_width_offset = -0.015

        if self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.02, 0.02)
            cube_y = 0.0 + self.np_random.uniform(-0.02, 0.02)
        else:
            cube_x, cube_y = 0.25, 0.0

        cube_half_h = float(self.model.geom_size[self._cube_geom_id][2])

        # Step 1: park the cube far away so the arm can settle without touching it
        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y + 0.5, cube_half_h]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Step 2: move arm to above-cube position with cube safely out of the way
        above_pos = np.array([cube_x, cube_y + finger_width_offset,
                               cube_half_h * 2 + grasp_z_offset + height_offset])
        locked_joints = [3, 4] if self.lock_wrist else []
        for _ in range(150):
            ctrl = self.ik.step_toward_target(above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=locked_joints)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        # Step 3: teleport cube into position now that the arm is clear
        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, cube_half_h]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        self.data.qvel[cube_qpos_addr - 1:cube_qpos_addr + 5] = 0  # zero cube velocity
        mujoco.mj_forward(self.model, self.data)

    def _reset_from_straddle_states(self):
        """Reset to a sampled qpos+ctrl from saved straddle-success snapshots + small noise."""
        idx = self.np_random.integers(0, len(self._straddle_states)) if self.np_random is not None else 0
        state = self._straddle_states[idx].copy()
        n_qpos = len(self.data.qpos)
        saved_qpos = state[:n_qpos]
        saved_ctrl = state[n_qpos:]
        # Add small noise to arm joints only (not cube free joint at the end)
        n_robot_dof = n_qpos - 7  # subtract cube free joint (3 pos + 4 quat)
        if self.np_random is not None:
            saved_qpos[:n_robot_dof] += self.np_random.uniform(-0.02, 0.02, size=n_robot_dof)
        self.data.qpos[:] = saved_qpos
        if len(saved_ctrl) == len(self.data.ctrl):
            self.data.ctrl[:] = saved_ctrl

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

        # Freeze cube in place: restore saved position and zero velocity
        if self.freeze_cube and self._frozen_cube_qpos is not None:
            cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
            qa = self.model.jnt_qposadr[cube_joint_id]
            da = self.model.jnt_dofadr[cube_joint_id]
            self.data.qpos[qa:qa + 7] = self._frozen_cube_qpos
            self.data.qvel[da:da + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        if info["is_lifted"]:
            self._hold_count += 1
        else:
            self._hold_count = 0
        info["hold_count"] = self._hold_count

        # Recompute is_success with updated hold_count (v_straddle/v_grip/v_zone have own logic)
        if "zone" in self.reward_version or "straddle" in self.reward_version:
            pass  # is_success already set correctly by _get_info
        elif "grip" in self.reward_version:
            info["is_success"] = info["is_grasping"]
        else:
            info["is_success"] = self._hold_count >= self.hold_steps

        reward = self._compute_reward(info, was_grasping=self._was_grasping, action=action)
        self._was_grasping = info["is_grasping"]
        self._prev_action = action.copy()

        # v_straddle and v_grip don't terminate on success — run to max_episode_steps
        # so the agent accumulates reward for holding the good state, not just touching it
        terminated = False  # always run to max_episode_steps
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
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
