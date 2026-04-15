"""Test how well a box replacement overlaps the moving jaw mesh's gripping face.

Shows:
- The mesh rendered from 4 angles
- A candidate box overlaid in transparent red
- Overlap % of the gripping face (inner contact surface) covered by the box
"""
import mujoco
import numpy as np
import cv2
from pathlib import Path

# ── Inline test scene: just the moving jaw mesh + a box candidate ─────────────
# Box params to tune: pos and size in moving_jaw body frame
# Gripping face is near x=-0.012, the arm runs along y axis
BOX_POS  = (-0.001, -0.033, 0.019)  # center of candidate box
BOX_SIZE = (0.012, 0.046, 0.021)    # half-sizes

XML = f"""
<mujoco>
  <compiler meshdir="models/so101/assets"/>
  <asset>
    <mesh name="jaw" file="moving_jaw_so101_v1.stl"/>
    <material name="mesh_mat" rgba="1 0.82 0.12 1"/>
    <material name="box_mat"  rgba="1 0 0 0.4"/>
    <material name="face_mat" rgba="0 0.8 0 0.8"/>
  </asset>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <!-- Mesh (visual only) -->
    <geom type="mesh" mesh="jaw" material="mesh_mat" pos="0 0 0.0189" contype="0" conaffinity="0"/>
    <!-- Box candidate (semi-transparent red) -->
    <geom type="box" size="{BOX_SIZE[0]} {BOX_SIZE[1]} {BOX_SIZE[2]}"
          pos="{BOX_POS[0]} {BOX_POS[1]} {BOX_POS[2]}"
          material="box_mat" contype="0" conaffinity="0"/>
    <!-- Gripping face marker: thin slab at the inner x face -->
    <geom type="box" size="0.0005 {BOX_SIZE[1]} {BOX_SIZE[2]}"
          pos="{-BOX_SIZE[0] + BOX_POS[0]} {BOX_POS[1]} {BOX_POS[2]}"
          material="face_mat" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

H, W = 400, 400

cameras = [
    # (name, pos, lookat, up)
    ("front (+X side)",   [0.15, 0, 0.02],   [0, -0.03, 0.02], [0,0,1]),
    ("inner face (-X)",   [-0.15, 0, 0.02],  [0, -0.03, 0.02], [0,0,1]),
    ("top (Z)",           [0, -0.03, 0.18],  [0, -0.03, 0.02], [0,1,0]),
    ("side (Y)",          [0, 0.18, 0.02],   [0, -0.03, 0.02], [0,0,1]),
]

renderer = mujoco.Renderer(model, height=H, width=W)
frames = []

for label, pos, lookat, up in cameras:
    # Set free camera
    renderer.update_scene(data)
    cam = renderer._scene.camera  # type: ignore
    # Use lookat camera via MjvCamera
    mjv_cam = mujoco.MjvCamera()
    mjv_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mjv_cam.lookat[:] = lookat
    mjv_cam.distance = float(np.linalg.norm(np.array(pos) - np.array(lookat)))
    # compute azimuth/elevation from pos-lookat vector
    diff = np.array(pos) - np.array(lookat)
    mjv_cam.azimuth   = float(np.degrees(np.arctan2(diff[1], diff[0])))
    mjv_cam.elevation = float(np.degrees(np.arcsin(diff[2] / (np.linalg.norm(diff) + 1e-8))))

    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=mjv_cam, scene_option=opt)
    img = renderer.render().copy()

    cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    frames.append(img)

renderer.close()

# ── Compute overlap on the gripping face ──────────────────────────────────────
import trimesh

mesh = trimesh.load("models/so101/assets/moving_jaw_so101_v1.stl")
# Shift mesh verts by the mesh pos offset in the XML (z+0.0189)
verts = mesh.vertices.copy()
verts[:, 2] += 0.0189

# Gripping face = faces whose normal points in -X and whose x coord is near mesh min_x
normals = mesh.face_normals
neg_x_mask = normals[:, 0] < -0.7
face_verts = verts[mesh.faces[neg_x_mask].flatten()]

# Box inner face: x = BOX_POS[0] - BOX_SIZE[0]
face_x = BOX_POS[0] - BOX_SIZE[0]
y_min, y_max = BOX_POS[1] - BOX_SIZE[1], BOX_POS[1] + BOX_SIZE[1]
z_min, z_max = BOX_POS[2] - BOX_SIZE[2], BOX_POS[2] + BOX_SIZE[2]

# For each -X face, check if its centroid is within box face bounds
face_centers = verts[mesh.faces[neg_x_mask]].mean(axis=1)
in_box = (
    (face_centers[:, 1] >= y_min) & (face_centers[:, 1] <= y_max) &
    (face_centers[:, 2] >= z_min) & (face_centers[:, 2] <= z_max)
)
total_area  = mesh.area_faces[neg_x_mask].sum()
covered_area = mesh.area_faces[neg_x_mask][in_box].sum()
overlap_pct = covered_area / total_area * 100 if total_area > 0 else 0.0

print(f"Gripping face (-X normal) faces: {neg_x_mask.sum()}")
print(f"Covered by box: {in_box.sum()} / {neg_x_mask.sum()} faces")
print(f"Area overlap: {overlap_pct:.1f}%")
print(f"Box inner face x={face_x:.4f},  y=[{y_min:.4f},{y_max:.4f}],  z=[{z_min:.4f},{z_max:.4f}]")
print(f"Mesh -X face x=[{face_verts[:,0].min():.4f},{face_verts[:,0].max():.4f}]  "
      f"y=[{face_verts[:,1].min():.4f},{face_verts[:,1].max():.4f}]  "
      f"z=[{face_verts[:,2].min():.4f},{face_verts[:,2].max():.4f}]")

# ── Assemble output image ─────────────────────────────────────────────────────
top = np.hstack(frames[:2])
bot = np.hstack(frames[2:])
out = np.vstack([top, bot])
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

info = (f"Box pos={BOX_POS}  size={BOX_SIZE}    "
        f"Gripping face overlap: {overlap_pct:.1f}%")
cv2.putText(out, info, (6, out.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

out_path = "jaw_box_fit.png"
cv2.imwrite(out_path, out)
print(f"\nSaved to {out_path}")
