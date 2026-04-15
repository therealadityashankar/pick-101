"""Visual check: render gripper at various opening angles with a cube placed between fingertips.

For each (cube_width, gripper_fraction) pair:
  - Places the cube centred between the two fingertip sites
  - Renders overview + wrist cam side by side
  - Labels the fingertip separation distance and whether the cube fits

Output: gripper_geometry_check.png

Usage:
    uv run python check_gripper_geometry.py
"""
import mujoco
import numpy as np
import cv2
from pathlib import Path

SCENE_PATH = "models/so101/lift_cube.xml"
OUTPUT_PATH = "gripper_geometry_check.png"

RENDER_H = 320
RENDER_W = 420

_GRIP_CLEARANCE = 0.005  # must match lift_cube_projection_t1v2.py

# Cube widths to visualise (mm). For each, we show:
#   - min fraction: exactly the computed minimum (should be snug)
#   - min + small delta: a few mm extra (should be clearly open)
CUBE_WIDTHS_MM = [20, 30, 40, 50, 60, 70]

# Camera for the overview panel — looking at the gripper from the side
CAM_LOOKAT   = [0.15, 0.0, 0.15]
CAM_DISTANCE = 0.30
CAM_AZIMUTH  = 160
CAM_ELEVATION = -15


def build_dist_table(model, data, n=300):
    """Sweep gripper joint, return (angles, inner_gap) arrays using pad geoms.

    Projects each pad's half-size onto the axis connecting the pad centres,
    matching the equation used in lift_cube_projection_t1v2._build_gripper_dist_table.
    """
    gid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    lo, hi = model.jnt_range[gid]
    addr   = model.jnt_qposadr[gid]
    sp_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    mp_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")

    angles    = np.linspace(lo, hi, n)
    distances = np.zeros(n)
    for i, a in enumerate(angles):
        data.qpos[addr] = a
        mujoco.mj_forward(model, data)
        sp = data.geom_xpos[sp_id].copy()
        mp = data.geom_xpos[mp_id].copy()
        diff = sp - mp
        dist = float(np.linalg.norm(diff))
        if dist < 1e-8:
            distances[i] = 0.0
            continue
        axis = diff / dist
        sp_mat = data.geom_xmat[sp_id].reshape(3, 3)
        mp_mat = data.geom_xmat[mp_id].reshape(3, 3)
        sp_half = abs(float(np.dot(sp_mat @ np.array([model.geom_size[sp_id][0], 0, 0]), axis)))
        mp_half = abs(float(np.dot(mp_mat @ np.array([model.geom_size[mp_id][0], 0, 0]), axis)))
        distances[i] = max(0.0, dist - sp_half - mp_half)
    return angles, distances, lo, hi, addr, sp_id, mp_id


def render_case(model, data, ov_renderer, wr_renderer,
                angles, distances, lo, hi, addr, sp_id, mp_id,
                cube_width_mm, gripper_frac):
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    cube_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_adr     = model.jnt_qposadr[cube_jnt_id]
    # Set gripper angle from fraction
    angle = lo + gripper_frac * (hi - lo)
    data.qpos[addr] = angle
    mujoco.mj_forward(model, data)

    # Measure inner gap between pad faces (axis-projected, matching env equation)
    sp   = data.geom_xpos[sp_id].copy()
    mp   = data.geom_xpos[mp_id].copy()
    diff = sp - mp
    dist = float(np.linalg.norm(diff))
    if dist > 1e-8:
        axis = diff / dist
        sp_mat = data.geom_xmat[sp_id].reshape(3, 3)
        mp_mat = data.geom_xmat[mp_id].reshape(3, 3)
        sp_half = abs(float(np.dot(sp_mat @ np.array([model.geom_size[sp_id][0], 0, 0]), axis)))
        mp_half = abs(float(np.dot(mp_mat @ np.array([model.geom_size[mp_id][0], 0, 0]), axis)))
        inner_gap = max(0.0, dist - sp_half - mp_half)
    else:
        inner_gap = 0.0

    # Place cube midpoint between pad centres
    mid_pos = (sp + mp) / 2.0

    # Resize cube geom
    half = cube_width_mm / 2 / 1000
    model.geom_size[cube_geom_id] = [half, half, half]

    # Move cube body: set freejoint qpos (pos + quaternion)
    data.qpos[cube_adr:cube_adr + 3]  = mid_pos
    data.qpos[cube_adr + 3:cube_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    fits = inner_gap >= cube_width_mm / 1000

    # Overview render
    cam = mujoco.MjvCamera()
    cam.lookat[:]  = CAM_LOOKAT
    cam.distance   = CAM_DISTANCE
    cam.azimuth    = CAM_AZIMUTH
    cam.elevation  = CAM_ELEVATION
    ov_renderer.update_scene(data, camera=cam)
    ov = cv2.cvtColor(ov_renderer.render().copy(), cv2.COLOR_RGB2BGR)

    # Wrist cam render
    wr_renderer.update_scene(data, camera="wrist_cam")
    wr_raw = wr_renderer.render().copy()
    # square-crop wrist cam
    crop_x = (RENDER_W - RENDER_H) // 2
    wr = cv2.cvtColor(wr_raw[:, crop_x:crop_x + RENDER_H], cv2.COLOR_RGB2BGR)
    wr = cv2.resize(wr, (RENDER_W, RENDER_H))

    # Labels
    color = (0, 200, 0) if fits else (0, 0, 220)
    status = "FITS" if fits else "TOO BIG"
    label1 = f"cube={cube_width_mm}mm  frac={gripper_frac:.2f}  [{status}]"
    label2 = f"gap={inner_gap*1000:.1f}mm  angle={np.degrees(angle):.1f}deg"
    for img in (ov, wr):
        cv2.putText(img, label1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(img, label2, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    div = np.full((RENDER_H, 4, 3), 50, dtype=np.uint8)
    return np.hstack([ov, div, wr])


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    ov_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    wr_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    angles, distances, lo, hi, addr, sp_id, mp_id = build_dist_table(model, data)
    sp_half = float(model.geom_size[sp_id][0])
    mp_half = float(model.geom_size[mp_id][0])

    print("Gripper geometry (inner gap between pad faces):")
    print(f"  Joint range: [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}] deg")
    print(f"  Gap range:   [{distances.min()*1000:.1f}, {distances.max()*1000:.1f}] mm\n")

    rows = []
    for cube_mm in CUBE_WIDTHS_MM:
        half    = cube_mm / 2 / 1000
        needed  = 2.0 * half * (1.0 + _GRIP_CLEARANCE)
        angle   = float(np.interp(needed, distances, angles))
        frac    = float(np.clip((angle - lo) / (hi - lo), 0.0, 1.0))

        # Verify: what inner gap do we get at that angle?
        data.qpos[addr] = angle
        mujoco.mj_forward(model, data)
        gap = max(0.0, float(np.linalg.norm(
            data.geom_xpos[sp_id] - data.geom_xpos[mp_id])) - sp_half - mp_half) * 1000
        print(f"  {cube_mm:>4}mm   needed={needed*1000:5.1f}mm   frac={frac:.3f}   gap@min={gap:.1f}mm")

        row = render_case(model, data, ov_renderer, wr_renderer,
                          angles, distances, lo, hi, addr, sp_id, mp_id,
                          cube_mm, frac)
        rows.append(row)

    ov_renderer.close()
    wr_renderer.close()

    out = np.vstack(rows)
    cv2.imwrite(OUTPUT_PATH, out)
    print(f"\nSaved to {OUTPUT_PATH}  ({out.shape[1]}×{out.shape[0]}px)")


if __name__ == "__main__":
    main()
