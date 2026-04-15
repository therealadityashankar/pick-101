"""Working reward functions for lift cube task.

These are the reward functions that achieve 100% success rate:
- v11: State-based (SAC) - 100% success at 1M steps
- v19: Image-based (DrQ-v2) - 100% success at 2M steps

For historical/experimental reward versions, see _legacy_rewards.py.
"""

import numpy as np


def reward_v11(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V11: Dense reward for state-based training.

    Structure:
    - Reach reward (tanh distance)
    - Push-down penalty
    - Drop penalty
    - Grasp bonus + continuous lift reward
    - Binary lift bonus
    - Target height bonus
    - Action rate penalty (only when lifted)
    - Success bonus

    Achieved 100% success at 1M steps with SAC.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
    if action is not None and cube_z > 0.06:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v_straddle(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """Stage 1 (straddle): descend around the cube without touching it.

    Phase 1 — Centering (inherited from zone):
        Keep the cube centered between the two fingertips in XY.  The gripper
        midpoint should align with the cube center, and the fingertips should
        be on opposite sides (straddling in XY).

    Phase 2 — Controlled descent:
        Reward the gripper for descending toward the cube top face.  Once the
        fingertips are *below* the cube top face, reward proximity to the cube
        sides without actually touching it.  A small gripper opening is
        encouraged during descent so the fingers clear the cube edges.

    Phase 3 — Depth limit:
        Penalise going too far below the cube center z (over-descent).

    Success: both fingertips below cube top face, on opposite sides, within
    6 cm XY, cube upright, no contact.
    """
    cube_pos = info["cube_pos"]
    cube_top_z = info["cube_top_z"]
    static_pos = info["static_fingertip_pos"]
    moving_pos = info["moving_fingertip_pos"]
    has_gripper_contact = info["has_gripper_contact"]
    has_jaw_contact = info["has_jaw_contact"]
    gripper_pos = info.get("gripper_pos", static_pos)

    reward = 0.0

    # ── 1. XY centering (same as zone) ──────────────────────────────────
    # Steer both fingertips toward the cube in XY so the cube stays centered
    # between them.  tanh(5·d) saturates around d≈0.06 m — beyond that the
    # gradient is nearly flat, so the agent is free to focus on Z.
    static_xy_dist = np.linalg.norm(static_pos[:2] - cube_pos[:2])
    moving_xy_dist = np.linalg.norm(moving_pos[:2] - cube_pos[:2])
    reward += (1.0 - np.tanh(5.0 * static_xy_dist)) * 2.0
    reward += (1.0 - np.tanh(5.0 * moving_xy_dist)) * 2.0

    # ── 2. Straddling in XY ─────────────────────────────────────────────
    # Fingertips must be on opposite sides of the cube along the finger axis.
    # static_side < 0 and moving_side > 0 means they straddle correctly.
    finger_vec = moving_pos - static_pos
    finger_axis = finger_vec / (np.linalg.norm(finger_vec) + 1e-8)
    static_side = np.dot(static_pos - cube_pos, finger_axis)
    moving_side = np.dot(moving_pos - cube_pos, finger_axis)
    straddle = np.clip(-static_side, 0, None) + np.clip(moving_side, 0, None)
    reward += np.tanh(10.0 * straddle) * 2.0

    # ── 3. Descent reward ───────────────────────────────────────────────
    # How far each fingertip is below the cube's top face.
    # static_below > 0 means fingertip is below the top face.
    static_below = cube_top_z - static_pos[2]
    moving_below = cube_top_z - moving_pos[2]

    # Descent bonus: gated by XY centering — only rewarded when already well-centered.
    # This prevents the exploit of going low anywhere in the workspace.
    xy_centered = (1.0 - np.tanh(5.0 * static_xy_dist)) * (1.0 - np.tanh(5.0 * moving_xy_dist))
    reward += max(0.0, np.tanh(20.0 * static_below)) * 1.5 * xy_centered
    reward += max(0.0, np.tanh(20.0 * moving_below)) * 1.5 * xy_centered

    # ── 4. Gripper opening during descent ───────────────────────────────
    # While above the cube top, encourage a slightly open gripper so the
    # fingers clear the cube edges.  We target min_gripper_frac (the minimum
    # opening that fits around the cube).  Below the top face we don't care
    # — the agent can start closing if it wants.
    gripper_state = info.get("gripper_state", 0.0)
    gripper_lo = getattr(env, "_gripper_lo", 0.0)
    gripper_hi = getattr(env, "_gripper_hi", 1.0)
    gripper_frac = float(np.clip(
        (gripper_state - gripper_lo) / (gripper_hi - gripper_lo + 1e-8), 0.0, 1.0
    ))
    min_frac = getattr(env, "_min_gripper_frac", 0.3)
    # Only penalise when fingertips are still above cube top (descent phase)
    avg_below = (static_below + moving_below) / 2.0
    if avg_below < 0:  # still above cube top
        # Only penalise being too closed (can't clear cube edges); too open is fine
        under_open = max(0.0, min_frac - gripper_frac)
        reward -= 1.0 * under_open

    # ── 5. Side proximity (once below top face, gated by centering) ───────
    # Once fingertips are below the cube top, reward them for being *close*
    # to the cube sides along the finger axis — ideally ~10 mm from the
    # cube face (cube_half + 10 mm clearance).
    if static_below > 0 and moving_below > 0:
        cube_half = env._cube_half_z  # cube is a cube, so half_z ≈ half_x
        ideal_clearance = cube_half + 0.010  # 10 mm gap beyond cube face
        static_finger_dist = abs(float(static_side))
        moving_finger_dist = abs(float(moving_side))
        static_prox = 1.0 - np.tanh(30.0 * abs(static_finger_dist - ideal_clearance))
        moving_prox = 1.0 - np.tanh(30.0 * abs(moving_finger_dist - ideal_clearance))
        reward += static_prox * 0.5 * xy_centered
        reward += moving_prox * 0.5 * xy_centered

    # ── 6. Over-descent penalty ─────────────────────────────────────────
    # Don't go more than ~1.5 cube heights below the cube center.
    # cube_pos[2] is cube center z; going below cube_pos[2] - cube_half_z
    # means the fingertips are below the cube bottom face.
    cube_bottom_z = cube_pos[2] - env._cube_half_z
    static_over = cube_bottom_z - static_pos[2]  # positive = below cube bottom
    moving_over = cube_bottom_z - moving_pos[2]
    if static_over > 0:
        reward -= 3.0 * static_over / env._cube_half_z
    if moving_over > 0:
        reward -= 3.0 * moving_over / env._cube_half_z

    # ── 7. Success bonus (gated by centering) ───────────────────────────
    if (static_below > 0 and moving_below > 0
            and static_side < 0.0 and moving_side > 0.0):
        reward += 5.0 * xy_centered

    return reward


def reward_v_grip(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T2: reward for grasping the cube.

    Lean grasp-only reward — assumes agent already knows how to straddle (transferred from T1).
    No straddle geometry shaping, just: reach + close + hold.
    Success = cube is grasped (both contacts + closed gripper).
    """
    reward = 0.0
    is_grasping = info["is_grasping"]

    # Reach reward — keep gripper near cube
    reach_reward = 1.0 - np.tanh(10.0 * info["gripper_to_cube"])
    reward += reach_reward

    if is_grasping:
        reward += 3.0

    if was_grasping and not is_grasping:
        reward -= 2.0

    if info["is_success"]:
        reward += 5.0

    return reward


def reward_v_lift(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T3: reward for lifting the cube after grasping.

    Assumes agent already knows how to grasp — focuses on lifting.
    Success = cube lifted to target height and held.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    is_grasping = info["is_grasping"]

    # Reach reward to keep gripper near cube
    reach_reward = 1.0 - np.tanh(10.0 * info["gripper_to_cube"])
    reward += reach_reward

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    if is_grasping:
        reward += 1.0
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 3.0

    if cube_z > env.lift_height:
        reward += 2.0
        reward += 0.5 * info["hold_count"]

    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v11_proj(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V11 + visibility bonus for projection-based policy.

    Adds a reward for keeping cube corners visible in the wrist camera frame.
    corners_in_frame is the fraction of the 4 top-face corners projecting within [0,1].
    """
    reward = reward_v11(env, info, was_grasping=was_grasping, action=action)

    # Bonus of cube corners are in frame
    corners_in_frame = info.get("corners_in_frame", 1.0)
    reward += 0.5 * corners_in_frame
    return reward


def reward_v_zone(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """Stage -1 (zoning): align the gripper over the cube in image space only.

    Uses projected 2D coordinates exclusively — no 3D state, no Z/depth signal.
    Goal: the projected cube center should sit at the midpoint between the two
    projected fingertips, and the fingertips should be on opposite sides of the
    cube center along the gripper axis.

    Does NOT reward/penalize Z movement — only XY centering in the image.
    Success: cube projected center within 5% of image width from gripper midpoint.
    """
    obs = env._get_obs()  # [su, sv, mu, mv, c0u, c0v, ..., c3u, c3v, gripper]
    su, sv = obs[0], obs[1]
    mu, mv = obs[2], obs[3]
    # Cube corners: obs[4:12] = 4 corners × (u, v)
    cube_corners = obs[4:12].reshape(4, 2)
    cube_center = cube_corners.mean(axis=0)  # projected cube center [u, v]
    cu, cv = float(cube_center[0]), float(cube_center[1])

    # Midpoint between the two fingertip projections
    grip_mid_u = (su + mu) / 2.0
    grip_mid_v = (sv + mv) / 2.0

    # Distance from gripper midpoint to cube projected center (in normalised [0,1] space)
    center_dist = float(np.sqrt((grip_mid_u - cu) ** 2 + (grip_mid_v - cv) ** 2))
    reward = (1.0 - np.tanh(15.0 * center_dist)) * 2.0

    # Bonus: fingertips should straddle the cube center along the gripper axis
    grip_vec = np.array([mu - su, mv - sv])
    grip_len = np.linalg.norm(grip_vec) + 1e-8
    grip_axis = grip_vec / grip_len
    static_side = np.dot(np.array([su - cu, sv - cv]), grip_axis)
    moving_side = np.dot(np.array([mu - cu, mv - cv]), grip_axis)
    straddle_2d = np.clip(-static_side, 0, None) + np.clip(moving_side, 0, None)
    reward += np.tanh(20.0 * straddle_2d) * 1.0

    # Keep cube corners visible in frame
    corners_in_frame = info.get("corners_in_frame", 1.0)
    reward += 0.3 * corners_in_frame

    # Gripper opening penalty — stay near min_gripper_frac, not wide open
    gripper_state = info.get("gripper_state", 0.0)
    gripper_lo = getattr(env, "_gripper_lo", 0.0)
    gripper_hi = getattr(env, "_gripper_hi", 1.0)
    gripper_frac = float(np.clip((gripper_state - gripper_lo) / (gripper_hi - gripper_lo + 1e-8), 0.0, 1.0))
    min_frac = getattr(env, "_min_gripper_frac", 0.3)
    gripper_error = abs(gripper_frac - min_frac)
    reward -= 1.5 * gripper_error

    # Z movement penalty — penalise downward gripper movement to avoid tipping
    gripper_pos = info.get("gripper_pos")
    if gripper_pos is not None:
        cube_top_z = info.get("cube_top_z", 0.05)
        clearance = gripper_pos[2] - cube_top_z
        if clearance < 0.04:  # getting too close vertically
            reward -= 3.0 * (0.04 - clearance)

    # Tipping penalty — cube should stay upright even in this stage
    cube_upright = info.get("cube_upright", 1.0)
    if cube_upright < 0.85:
        reward -= 3.0 * (1.0 - cube_upright)

    # Cube velocity penalty — don't knock it
    cube_vel = info.get("cube_vel", 0.0)
    if cube_vel > 0.01:
        reward -= 2.0 * cube_vel

    # Success bonus
    if center_dist < 0.10 and static_side < 0.0 and moving_side > 0.0:
        reward += 3.0

    return reward


def reward_v_unified_zone(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T1v2 stage -1 (zoning): 2D image-space centering before navigation."""
    return reward_v_zone(env, info, was_grasping=was_grasping, action=action)


def reward_v_unified_straddle(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T1v2 stage 0: straddle with domain randomisation obs (cube size + FOV).
    Delegates to v_straddle; stage obs is pinned to 0.0 by the env.
    """
    return reward_v_straddle(env, info, was_grasping=was_grasping, action=action)


def reward_v_unified_grip(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T1v2 stage 1: grip with domain randomisation obs.
    Delegates to v_grip; stage obs is pinned to 0.5 by the env.
    """
    return reward_v_grip(env, info, was_grasping=was_grasping, action=action)


def reward_v_unified_lift(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T1v2 stage 2: lift with domain randomisation obs.
    Delegates to v_lift; stage obs is pinned to 1.0 by the env.
    """
    return reward_v_lift(env, info, was_grasping=was_grasping, action=action)


def reward_v_unified(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """T1.1 unified reward: dispatches to stage-appropriate sub-reward.

    Stage 0 (navigate/straddle): v_straddle
    Stage 1 (grip):               v_grip
    Stage 2 (lift):               v_lift

    Transition bonuses are applied in the env's step() method.
    """
    stage = info.get("stage", 0)
    if stage == 0:
        return reward_v_straddle(env, info, was_grasping=was_grasping, action=action)
    elif stage == 1:
        return reward_v_grip(env, info, was_grasping=was_grasping, action=action)
    else:
        return reward_v_lift(env, info, was_grasping=was_grasping, action=action)


def reward_v19(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V19: Dense reward for image-based training.

    Key innovations over v11:
    - Per-finger reach reward (moving finger gets own reach gradient)
    - Stronger grasp bonus (1.5 vs 0.25)
    - Doubled lift coefficient (4.0 vs 2.0)
    - Threshold ramp from 0.04m to 0.08m
    - Hold count bonus (escalating reward for sustained height)

    Achieved 100% success at 2M steps with DrQ-v2.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    gripper_state = info["gripper_state"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    is_closed = gripper_state < 0.25

    # Standard gripper reach (static finger is part of gripper frame)
    gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_cube)

    # Moving finger reach - only applies when gripper is close to cube
    reach_threshold = 0.7  # ~3cm from cube
    if gripper_reach < reach_threshold:
        reach_reward = gripper_reach
    else:
        if is_closed:
            moving_reach = 1.0
        else:
            moving_finger_pos = env._get_moving_finger_pos()
            moving_to_cube = np.linalg.norm(moving_finger_pos - cube_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_cube)

        reach_reward = (gripper_reach + moving_reach) * 0.5

    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 1.5

        # Continuous lift reward (4.0x coefficient)
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 4.0

        # Binary lift bonus at 0.02m
        if cube_z > 0.02:
            reward += 1.0

        # Linear threshold ramp from 0.04m to 0.08m
        if cube_z > 0.04:
            threshold_progress = min(1.0, (cube_z - 0.04) / (env.lift_height - 0.04))
            reward += threshold_progress * 2.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

        # Hold count bonus - escalating reward for sustained height
        reward += 0.5 * hold_count

    # Action rate penalty during hold phase
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward
