# pick-101

RL training and real-robot deployment for the SO-101 arm — picks up a Jenga block using a projection-based policy trained entirely in simulation.

https://github.com/user-attachments/assets/real_t1_run.mp4

## Installation

```bash
git clone git@github.com:ggand0/pick-101.git
cd pick-101
git submodule update --init --recursive
uv sync
```

Assets (STL meshes) are stored via Git LFS and pulled automatically. Alternatively, copy from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100):
```bash
cp -r SO-ARM100/Simulation/SO101/assets models/so101/
```

---

## Running on the Real Robot

### Prerequisites

- SO-101 arm connected via USB
- Printed calibration board (`make_aruco_board.py`) — flat on the table
- Printed Jenga block tag (`make_jenga_tag.py`) — ID 101, attached to top face of Jenga block
- Overhead camera pointing down at the board
- Robot base tip positioned at the bottom-centre of the board

### Step 1 — Print the board and block tag

```bash
# Calibration board (place flat on table, robot base at bottom-centre)
uv run python make_aruco_board.py        # → aruco_board.pdf

# Jenga block tag — ID 101, sized to fit the 25×75mm block face (use this one)
uv run python make_jenga_tag.py          # → jenga_tag.pdf

# (Optional) Full sheet of general-purpose bordered tags IDs 100–150
uv run python make_bordered_tags.py      # → bordered_tags.pdf
```

Print `jenga_tag.pdf` at **100% scale (no fit-to-page scaling)** and attach to the top face of the Jenga block.

### Step 2 — Restore the trained model (after fresh clone)

The best trained model is stored in `best_run/` and committed to git. Restore it into `runs/` so the scripts can find it:

```bash
uv run python export_best_run.py restore
```

### Step 3 — Calibrate joint mapping (first time or after hardware change)

Maps real robot joint readings to simulation joint angles. Produces `calibration/joint_calibration.json`.

```bash
# Full calibration (all 6 joints)
uv run python calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441

# Single joint only (e.g. after re-mounting the wrist)
uv run python calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441 --joint wrist_roll
```

**Controls during calibration:**
- Move the arm by hand to match the reference image shown
- `SPACE` — record this position
- `S` — skip this point
- `Q` — quit

### Step 4 — Calibrate block position detection

Corrects for camera angle so the detected block position matches its true location on the board.

```bash
uv run python tests/test_aruco_homography_3d.py
```

1. Press **C** to start calibration
2. Place the Jenga block (tag ID 101) at each of the 4 interior corners in order: **TL → TR → BL → BR**
   - The block's physical corner should touch the interior corner each time
   - Keep block orientation consistent across all 4 corners
3. Press **SPACE** at each position to record it
4. After 4 corners, the script prints `DELTA_X` / `DELTA_Y` values

Paste the printed values into the `DELTA_X` / `DELTA_Y` constants at the top of `run_real_t1.py` and `visualize_irl_block.py`.

**Fine-tuning manually:**

If the corrected position (green dot in the rectified panel) is still slightly off, adjust `DELTA_X` / `DELTA_Y` (mm) directly:

| Green dot position | Adjustment |
|--------------------|------------|
| Too far right | decrease `DELTA_X` |
| Too far left | increase `DELTA_X` |
| Too far down | decrease `DELTA_Y` |
| Too far up | increase `DELTA_Y` |

Copy the final values into `run_real_t1.py` and `visualize_irl_block.py`.

### Step 5 — Test block detection (no robot required)

```bash
uv run python visualize_irl_block.py --camera 0
```

Place the Jenga block on the board and verify the 4 panels:

| Panel | Shows |
|-------|-------|
| 1 — Camera | Raw camera feed with detected tags |
| 2 — Sim side | Simulation side view with block at detected position |
| 3 — Top-down (homography) | Rectified board view; green dot = corrected block position |
| 4 — Sim top-down | Simulation top-down; block should match panel 3 |

### Step 6 — Dry run (no robot)

```bash
uv run python run_real_t1.py --no-robot --camera 0
```

Verify all 5 panels look correct before connecting the robot.

### Step 7 — Run the policy on the real robot

```bash
uv run python run_real_t1.py --port /dev/tty.usbmodem5A680089441
```

Video is saved to `real_t1_run.mp4`.

---

## Exporting / Sharing the Trained Model

`best_run/` is committed to git. `runs/` is in `.gitignore`.

```bash
# Export best model from runs/ into best_run/ (then commit)
uv run python export_best_run.py export

# Restore best_run/ back into runs/ (after fresh clone)
uv run python export_best_run.py restore
```

---

## Training

### Projection-based policy (T1 — what runs on the real robot)

```bash
uv run python train_lift_projection.py --config configs/state_based/curriculum_stage3.yaml
```

### Evaluate a checkpoint

```bash
uv run python eval_projection.py --run runs/lift_proj_t1_s3/<timestamp>
```

### State-based curriculum (SAC)

```bash
uv run python train_lift.py --config configs/state_based/curriculum_stage3.yaml
uv run python eval_cartesian.py --run runs/lift_curriculum_s3/<timestamp>
```

### Plot learning curves

```bash
uv run python plot_learning_curves.py --run runs/lift_proj_t1_s3/<timestamp>
```

---

## Project Structure

```
models/so101/                   # MuJoCo robot models
├── lift_cube.xml               # Main scene
├── so101_new_calib.xml         # Robot with finger pads
└── assets/                     # STL meshes (Git LFS)

src/
├── envs/
│   ├── lift_cube.py            # Cartesian gym environment
│   └── lift_cube_projection.py # Projection-obs environment (used by T1 policy)
├── robot/
│   ├── real_robot.py           # SO-101 hardware interface
│   └── joint_calibration.py   # Real→sim joint angle mapping
└── training/
    └── train_image_rl.py       # DrQ-v2 image-based training

configs/state_based/            # Training configs

calibration/                    # Generated after running Step 3
└── joint_calibration.json

best_run/                       # Committed trained model
├── best_model/best_model.zip
├── vec_normalize.pkl
└── config.yaml

# Key scripts (root)
run_real_t1.py                  # Real-robot runner (Steps 6–7)
visualize_irl_block.py          # Block detection test (Step 5)
tests/test_aruco_homography_3d.py  # Homography viewer + position calibration (Step 4)
calibrate_joints_real.py        # Joint mapping calibration (Step 3)
export_best_run.py              # Export / restore trained model (Step 2)
make_aruco_board.py             # Generate printable calibration board (Step 1)
make_jenga_tag.py               # Generate printable Jenga block tag ID 101 (Step 1)
make_bordered_tags.py           # Generate full sheet of general-purpose bordered tags
train_lift_projection.py        # Train T1 policy
eval_projection.py              # Evaluate T1 policy
```

---

## Calibration Board Layout

The board is a 180×180mm square with ArUco border tags. The interior working area is 112×112mm. The robot base tip sits at the bottom-centre of the board.

```
┌─────────────────────────┐  ← 180mm
│   [border ArUco tags]   │
│  ┌───────────────────┐  │
│  │                   │  │
│  │   interior area   │  │
│  │   112 × 112 mm    │  │
│  │                   │  │
│  └───────────────────┘  │
│      ▲ robot base ▲     │
└─────────────────────────┘
```
