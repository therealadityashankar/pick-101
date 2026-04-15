"""Measure true inner gripper gap (between pad faces) vs joint angle.

Places a cube between the pads at each angle for visual verification.
Fits a model and outputs the inverse lookup (cube width → min fraction).

Usage:
    uv run python calibrate_gripper_gap.py
"""
import cv2
import mujoco
import numpy as np

SCENE_PATH = "models/so101/lift_cube.xml"
N_ANGLES = 30
RENDER_H = 280
RENDER_W = 370
GRID_COLS = 6


def measure_inner_gap(model, data, addr, angle, sp_id, mp_id):
    """Measure the gap between the inner faces of the two finger pads."""
    data.qpos[addr] = angle
    mujoco.mj_forward(model, data)

    sp = data.geom_xpos[sp_id].copy()
    mp = data.geom_xpos[mp_id].copy()
    diff = sp - mp
    dist = np.linalg.norm(diff)
    if dist < 1e-8:
        return 0.0, (sp + mp) / 2.0

    axis = diff / dist
    # Project each pad's half-size onto the connecting axis
    sp_mat = data.geom_xmat[sp_id].reshape(3, 3)
    mp_mat = data.geom_xmat[mp_id].reshape(3, 3)
    sp_half = abs(np.dot(sp_mat @ np.array([model.geom_size[sp_id][0], 0, 0]), axis))
    mp_half = abs(np.dot(mp_mat @ np.array([model.geom_size[mp_id][0], 0, 0]), axis))

    inner_gap = max(0.0, dist - sp_half - mp_half)
    mid_pos = (sp + mp) / 2.0
    return inner_gap, mid_pos


def render_panel(model, data, renderer, addr, angle, frac,
                 sp_id, mp_id, cube_geom_id, cube_qpos_addr,
                 inner_gap):
    """Render gripper with a cube sized to exactly the inner gap."""
    mujoco.mj_resetData(model, data)
    data.qpos[addr] = angle
    mujoco.mj_forward(model, data)

    sp = data.geom_xpos[sp_id].copy()
    mp = data.geom_xpos[mp_id].copy()
    mid_pos = (sp + mp) / 2.0

    # Size cube to exactly the inner gap (should be snug)
    half = inner_gap / 2.0
    if half < 0.001:
        half = 0.001
    model.geom_size[cube_geom_id] = [half, half, half]
    data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = mid_pos
    data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = mid_pos
    cam.distance = 0.18
    cam.azimuth = 160
    cam.elevation = -10
    renderer.update_scene(data, camera=cam)
    img = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

    cv2.putText(img, f"frac={frac:.2f}  gap={inner_gap*1000:.0f}mm",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"angle={np.degrees(angle):.0f}deg  cube={inner_gap*1000:.0f}mm",
                (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    gripper_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    lo, hi = model.jnt_range[gripper_jid]
    addr = model.jnt_qposadr[gripper_jid]

    sp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
    mp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_addr = model.jnt_qposadr[cube_jnt_id]

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    angles = np.linspace(lo, hi, N_ANGLES)
    gaps = np.zeros(N_ANGLES)
    panels = []

    print(f"Pad-based inner gap measurement")
    print(f"Joint range: [{np.degrees(lo):.1f}, {np.degrees(hi):.1f}] deg\n")
    print(f"{'angle_deg':>10}  {'frac':>6}  {'inner_gap_mm':>13}")
    print("-" * 36)

    for i, angle in enumerate(angles):
        mujoco.mj_resetData(model, data)
        frac = (angle - lo) / (hi - lo)
        gap, _ = measure_inner_gap(model, data, addr, angle, sp_id, mp_id)
        gaps[i] = gap
        print(f"  {np.degrees(angle):8.1f}°  {frac:5.3f}  {gap*1000:11.1f}mm")

        panel = render_panel(model, data, renderer, addr, angle, frac,
                             sp_id, mp_id, cube_geom_id, cube_qpos_addr, gap)
        panels.append(panel)

    renderer.close()

    # Assemble grid
    n_rows = (len(panels) + GRID_COLS - 1) // GRID_COLS
    blank = np.full((RENDER_H, RENDER_W, 3), 30, dtype=np.uint8)
    while len(panels) < n_rows * GRID_COLS:
        panels.append(blank)
    grid_rows = []
    for r in range(n_rows):
        grid_rows.append(np.hstack(panels[r * GRID_COLS:(r + 1) * GRID_COLS]))
    grid_img = np.vstack(grid_rows)
    cv2.imwrite("gripper_gap_visual.png", grid_img)
    print(f"\nVisual grid saved to gripper_gap_visual.png ({grid_img.shape[1]}x{grid_img.shape[0]}px)")

    # --- Fit ---
    from scipy.optimize import curve_fit

    def linear_model(angle, a, b):
        return a * angle + b

    lin_popt, _ = curve_fit(linear_model, angles, gaps)
    lin_rmse = np.sqrt(np.mean((gaps - linear_model(angles, *lin_popt))**2))

    print(f"\n--- Fit ---")
    print(f"Linear: gap = {lin_popt[0]*1000:.4f} * angle_rad + {lin_popt[1]*1000:.4f}  (mm)")
    print(f"  RMSE: {lin_rmse*1000:.3f} mm")

    # --- Inverse lookup + per-cube-size sanity images ---
    print(f"\n--- Cube width -> min gripper fraction ---")
    print(f"{'cube_mm':>8}  {'min_frac':>8}  {'angle_deg':>10}")
    print("-" * 32)

    cube_sizes = list(np.linspace(20, 52, 50))
    cube_panels = []
    renderer2 = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    for cube_mm in cube_sizes:
        cube_width = cube_mm / 1000.0
        if cube_width > gaps.max():
            print(f"  {cube_mm:>6.1f}mm   IMPOSSIBLE (max gap={gaps.max()*1000:.0f}mm)")
            cube_panels.append(np.full((RENDER_H, RENDER_W, 3), 30, dtype=np.uint8))
            continue
        angle_needed = float(np.interp(cube_width, gaps, angles))
        frac = (angle_needed - lo) / (hi - lo)
        print(f"  {cube_mm:>6.1f}mm   {frac:.3f}    {np.degrees(angle_needed):8.1f}°")

        # Render at min fraction with that cube size
        mujoco.mj_resetData(model, data)
        data.qpos[addr] = angle_needed
        mujoco.mj_forward(model, data)
        sp = data.geom_xpos[sp_id].copy()
        mp = data.geom_xpos[mp_id].copy()
        mid_pos = (sp + mp) / 2.0

        half = cube_mm / 2 / 1000
        model.geom_size[cube_geom_id] = [half, half, half]
        data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = mid_pos
        data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        cam = mujoco.MjvCamera()
        cam.lookat[:] = mid_pos
        cam.distance = 0.18
        cam.azimuth = 160
        cam.elevation = -10
        renderer2.update_scene(data, camera=cam)
        img = cv2.cvtColor(renderer2.render().copy(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"cube={cube_mm:.1f}mm  frac={frac:.3f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"angle={np.degrees(angle_needed):.1f}deg",
                    (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cube_panels.append(img)

    renderer2.close()

    cols = 10
    n_rows = (len(cube_panels) + cols - 1) // cols
    blank = np.full((RENDER_H, RENDER_W, 3), 30, dtype=np.uint8)
    while len(cube_panels) < n_rows * cols:
        cube_panels.append(blank)
    cube_grid = np.vstack([np.hstack(cube_panels[r*cols:(r+1)*cols]) for r in range(n_rows)])
    cv2.imwrite("gripper_cube_sizes.png", cube_grid)
    print(f"\nPer-cube-size sanity image saved to gripper_cube_sizes.png")

    # --- Save plot ---
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fracs = (angles - lo) / (hi - lo)
        ax.plot(fracs, gaps * 1000, 'bo-', markersize=5, label='measured (pad inner gap)')
        ax.plot(fracs, linear_model(angles, *lin_popt) * 1000, 'r--',
                label=f'linear fit (RMSE={lin_rmse*1000:.2f}mm)')
        ax.set_xlabel('Gripper fraction (0=closed, 1=open)')
        ax.set_ylabel('Inner gap between pads (mm)')
        ax.set_title('Gripper pad inner gap vs fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for cube_mm in [20, 30, 40, 50, 60, 70]:
            ax.axhline(cube_mm, color='gray', linestyle=':', alpha=0.5)
            ax.text(0.01, cube_mm + 1, f'{cube_mm}mm', fontsize=8, color='gray')
        plt.tight_layout()
        plt.savefig('gripper_gap_calibration.png', dpi=150)
        print(f"\nPlot saved to gripper_gap_calibration.png")
    except ImportError:
        print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
