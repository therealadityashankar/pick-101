"""Side-by-side visual comparison: original vs simplified robot model.

Renders both models from the same camera angle and saves to simplified_robot_check.png.

Usage:
    python check_simplified_robot.py
"""
import mujoco
import numpy as np
import cv2
from pathlib import Path

ORIGINAL_XML   = "models/so101/lift_cube.xml"
SIMPLIFIED_XML = "models/so101/lift_cube_simplified.xml"
OUTPUT_PATH    = "simplified_robot_check.png"

RENDER_H = 480
RENDER_W = 640


POSES = [
    dict(title="neutral",   joints={"shoulder_pan": 0,           "shoulder_lift": 0,          "elbow_flex": 0,          "wrist_flex": np.pi/2, "wrist_roll": np.pi/2}),
    dict(title="reaching",  joints={"shoulder_pan": 0,           "shoulder_lift": -np.pi/4,   "elbow_flex": np.pi/3,    "wrist_flex": np.pi/2, "wrist_roll": np.pi/2}),
    dict(title="side-reach",joints={"shoulder_pan": np.pi/4,     "shoulder_lift": -np.pi/4,   "elbow_flex": np.pi/4,    "wrist_flex": np.pi/2, "wrist_roll": 0}),
    dict(title="low-grasp", joints={"shoulder_pan": 0,           "shoulder_lift": -np.pi/2,   "elbow_flex": np.pi/2,    "wrist_flex": np.pi/4, "wrist_roll": np.pi/2}),
]

CAM = dict(lookat=[0.15, 0.0, 0.18], distance=0.7, azimuth=135, elevation=-20)


def render_model(xml_path: str, label: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    overview_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    wrist_renderer    = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM["lookat"]
    cam.distance   = CAM["distance"]
    cam.azimuth    = CAM["azimuth"]
    cam.elevation  = CAM["elevation"]

    rows = []
    for pose in POSES:
        mujoco.mj_resetData(model, data)
        for jname, angle in pose["joints"].items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = angle
        mujoco.mj_forward(model, data)

        # Overview panel
        overview_renderer.update_scene(data, camera=cam)
        overview = cv2.cvtColor(overview_renderer.render().copy(), cv2.COLOR_RGB2BGR)
        cv2.putText(overview, f"{label} — {pose['title']}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Wrist cam panel (square crop)
        wrist_renderer.update_scene(data, camera="wrist_cam")
        wrist_raw = wrist_renderer.render().copy()
        crop_x = (RENDER_W - RENDER_H) // 2
        wrist = cv2.cvtColor(wrist_raw[:, crop_x:crop_x + RENDER_H], cv2.COLOR_RGB2BGR)
        wrist = cv2.resize(wrist, (RENDER_W, RENDER_H))
        cv2.putText(wrist, "wrist cam", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        divider = np.full((RENDER_H, 4, 3), 60, dtype=np.uint8)
        rows.append(np.hstack([overview, divider, wrist]))

    overview_renderer.close()
    wrist_renderer.close()
    return np.vstack(rows)


def main():
    orig_col = render_model(ORIGINAL_XML,   "Original")
    simp_col = render_model(SIMPLIFIED_XML, "Simplified")

    divider = np.full((orig_col.shape[0], 6, 3), 60, dtype=np.uint8)
    combined = np.hstack([orig_col, divider, simp_col])

    cv2.imwrite(OUTPUT_PATH, combined)
    print(f"Saved to {OUTPUT_PATH}  ({combined.shape[1]}×{combined.shape[0]}px)")


if __name__ == "__main__":
    main()
