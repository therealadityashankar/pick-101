"""T1.1: Unified multi-stage projection policy with domain randomisation.

Extends LiftCubeProjectionEnv with:
  - Variable cube size per episode  (cube_size_range)
  - FOV randomisation per episode   (fov_noise)
  - Gripper-aware obs: pixel distance between fingertips, min opening fraction
  - Stage obs: navigate(0.0) → grip(0.5) → lift(1.0)
  - Unified reward dispatching to stage-appropriate sub-rewards

Obs dim: 16
  [0:4]    static_x, static_y, moving_x, moving_y  (fingertip projections)
  [4:12]   cube top-face corners c0..c3 xy           (4 corners × 2)
  [12]     gripper_opening                           (normalised)
  [13]     gripper_spread                          (pixel dist between fingertips, normalised)
  [14]     min_gripper_frac                          (fraction of range needed to clear cube)
  [15]     stage                                     (0.0 / 0.5 / 1.0)
"""
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.envs.rewards import REWARD_FUNCTIONS

# Clearance margin added on top of cube width when computing min opening angle
_GRIP_CLEARANCE = 0.005  # 0.5 % wider than cube
# Stage encoding values
_STAGE_VALS = {-1: 0.0, 0: 0.33, 1: 0.67, 2: 1.0}


class LiftCubeProjectionT1V2Env(LiftCubeProjectionEnv):
    """Unified three-stage pick-and-lift with domain randomisation."""

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 350,
        action_scale: float = 0.02,
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "dense",
        reward_version: str = "v_unified",
        curriculum_stage: int = 3,
        lock_wrist: bool = True,
        cam_pos_noise: float = 0.005,
        cam_euler_noise: float = 0.06,
        obs_noise: float = 0.01,
        straddle_states_path: str | None = None,
        freeze_cube: bool = False,
        # T1.1 extensions
        cube_size_range: tuple[float, float] = (0.011, 0.019),  # half-size metres (width/depth)
        cube_height_range: tuple[float, float] = (0.010, 0.035),  # half-size metres (height)
        fov_noise: float = 7.0,   # ±degrees randomisation around nominal fovy
    ):
        # Force reward_version so base class _compute_reward dispatches correctly
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            action_scale=action_scale,
            lift_height=lift_height,
            hold_steps=hold_steps,
            reward_type=reward_type,
            reward_version=reward_version,
            curriculum_stage=curriculum_stage,
            lock_wrist=lock_wrist,
            cam_pos_noise=cam_pos_noise,
            cam_euler_noise=cam_euler_noise,
            obs_noise=obs_noise,
            straddle_states_path=straddle_states_path,
            freeze_cube=freeze_cube,
        )

        self.cube_size_range = cube_size_range
        self.cube_height_range = cube_height_range
        self.fov_noise = fov_noise
        self._nominal_fovy = float(self.model.cam_fovy[self._wrist_cam_id])
        self._current_cube_half_size = 0.015  # updated each reset
        self._min_gripper_frac = 0.5          # updated each reset
        self._stage = {"v_unified_zone": -1, "v_unified_straddle": 0, "v_unified_grip": 1, "v_unified_lift": 2}.get(self.reward_version, 0)

        # Precompute gripper joint angle → fingertip distance table
        self._gripper_angles, self._gripper_dists = self._build_gripper_dist_table()

        # Override obs space: 16-dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gripper geometry table
    # ------------------------------------------------------------------

    def _build_gripper_dist_table(self, n: int = 300):
        """Sweep gripper joint, return (angles, inner_gap) arrays.

        Measures the gap between the inner faces of the finger pads by
        projecting each pad's half-size onto the axis connecting their centres.
        This gives the true usable clearance between the gripping surfaces.
        """
        data_tmp = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data_tmp)
        gripper_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        lo, hi = self.model.jnt_range[gripper_jid]
        addr = self.model.jnt_qposadr[gripper_jid]
        angles = np.linspace(lo, hi, n)
        dists = np.zeros(n)
        for i, a in enumerate(angles):
            data_tmp.qpos[addr] = a
            mujoco.mj_forward(self.model, data_tmp)
            sp = data_tmp.geom_xpos[self._static_pad_geom_id].copy()
            mp = data_tmp.geom_xpos[self._moving_pad_geom_id].copy()
            diff = sp - mp
            dist = float(np.linalg.norm(diff))
            if dist < 1e-8:
                dists[i] = 0.0
                continue
            axis = diff / dist
            sp_mat = data_tmp.geom_xmat[self._static_pad_geom_id].reshape(3, 3)
            mp_mat = data_tmp.geom_xmat[self._moving_pad_geom_id].reshape(3, 3)
            sp_half = abs(float(np.dot(
                sp_mat @ np.array([self.model.geom_size[self._static_pad_geom_id][0], 0, 0]), axis)))
            mp_half = abs(float(np.dot(
                mp_mat @ np.array([self.model.geom_size[self._moving_pad_geom_id][0], 0, 0]), axis)))
            dists[i] = max(0.0, dist - sp_half - mp_half)
        return angles, dists

    def _min_gripper_angle_for_cube(self, cube_half_size: float) -> float:
        """Return the gripper joint angle (rad) needed to clear the cube."""
        needed = 2.0 * cube_half_size * (1.0 + _GRIP_CLEARANCE)
        return float(np.interp(needed, self._gripper_dists, self._gripper_angles))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()  # 13-dim

        # Gripper pixel width: normalised pixel distance between fingertip projections
        K, R, t = self._get_camera_matrices()
        static_pos = self.data.site_xpos[self._static_fingertip_site_id].reshape(1, 3)
        moving_pos = self.data.site_xpos[self._moving_fingertip_site_id].reshape(1, 3)
        sp_norm = self._project_points_norm(static_pos, K, R, t)[0]
        mp_norm = self._project_points_norm(moving_pos, K, R, t)[0]
        gripper_spread = float(np.clip(np.linalg.norm(sp_norm - mp_norm), 0.0, 1.0))

        stage_val = float(_STAGE_VALS[self._stage])

        obs = np.append(base_obs, [gripper_spread, self._min_gripper_frac, stage_val])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Camera noise (also randomises FOV)
    # ------------------------------------------------------------------

    def _apply_camera_noise(self):
        super()._apply_camera_noise()
        if self.fov_noise > 0.0 and self.np_random is not None:
            delta_fov = self.np_random.uniform(-self.fov_noise, self.fov_noise)
        else:
            delta_fov = 0.0
        self.model.cam_fovy[self._wrist_cam_id] = self._nominal_fovy + delta_fov

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Randomise cube size BEFORE super().reset() so _reset_gripper_near_cube
        # reads the correct height when positioning the arm above the cube.
        # np_random is seeded by super().__init__ so it's available here.
        if self.np_random is not None:
            hs = float(self.np_random.uniform(*self.cube_size_range))
            hh = float(self.np_random.uniform(*self.cube_height_range))
        else:
            hs = 0.015
            hh = 0.015
        hh = min(hh, 2.0 * hs)  # cap height at 2× width to avoid unstable towers
        self._current_cube_half_size = hs
        self.model.geom_size[self._cube_geom_id] = [hs, hs, hh]
        self._cube_half_z = hh  # height, used for top-face projection and success check

        obs, info = super().reset(seed=seed, options=options)

        mujoco.mj_forward(self.model, self.data)

        # Compute min gripper fraction
        lo, hi = self._gripper_lo, self._gripper_hi
        angle = self._min_gripper_angle_for_cube(hs)
        self._min_gripper_frac = float(np.clip((angle - lo) / (hi - lo), 0.0, 1.0))

        self._stage = {"v_unified_zone": -1, "v_unified_straddle": 0, "v_unified_grip": 1, "v_unified_lift": 2}.get(self.reward_version, 0)

        # Return fresh obs (with updated cube geom)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------
    # Step (stage transitions + unified termination)
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        # Propagate stage into info so reward function can read it
        info["stage"] = self._stage

        # Stage is pinned for the whole episode (set at reset from reward_version)
        # Overall success only counts in lift stage
        if self._stage == 2:
            terminated = info["is_success"]
        else:
            terminated = False

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Info (override is_success for stage-aware semantics)
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info["stage"] = self._stage
        info["cube_half_size"] = self._current_cube_half_size
        # is_success is already set correctly by the base class per reward_version
        # (v_straddle sets it from straddle geometry, v_grip from grasp, v_lift from hold)
        return info
