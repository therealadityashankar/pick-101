# pick-101 — IK Controller

Inverse Kinematics controller for the SO-101 arm with vision-based calibration and real-robot deployment.

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

### Step 2 — Calibrate joint mapping (first time or after hardware change)

Maps real robot joint readings to simulation joint angles. Produces `.calibration/joint_calibration.json`.

```bash
# Full calibration (all 6 joints)
uv run python calibration/calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441

# Single joint only (e.g. after re-mounting the wrist)
uv run python calibration/calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441 --joint wrist_roll
```

**Controls during calibration:**
- Move the arm by hand to match the reference image shown
- `SPACE` — record this position
- `S` — skip this point
- `Q` — quit

### Step 3 — Calibrate block position detection

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

Paste the printed values into the `DELTA_X` / `DELTA_Y` constants at the top of `run_real_ik.py` and `visualize_irl_block.py`.

**Fine-tuning manually:**

If the corrected position (green dot in the rectified panel) is still slightly off, adjust `DELTA_X` / `DELTA_Y` (mm) directly:

| Green dot position | Adjustment |
|--------------------|------------|
| Too far right | decrease `DELTA_X` |
| Too far left | increase `DELTA_X` |
| Too far down | decrease `DELTA_Y` |
| Too far up | increase `DELTA_Y` |

Copy the final values into `run_real_ik.py` and `visualize_irl_block.py`.

### Step 4 — Test block detection (no robot required)

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

### Step 5 — Dry run (no robot)

```bash
uv run python run_real_ik.py --no-robot --camera 0
```

Verify all 5 panels look correct before connecting the robot.

### Step 6 — Run the IK controller on the real robot

```bash
uv run python run_real_ik.py --port /dev/tty.usbmodem5A680089441
```

Video is saved to `real_ik_run.mp4`.

---

## Project Structure

```
models/so101/                   # MuJoCo robot models
├── lift_cube.xml               # Main scene with calibration board
└── assets/                     # STL meshes (Git LFS)

src/
├── controllers/
│   └── ik_controller.py        # Damped least-squares IK solver
├── envs/
│   ├── lift_cube.py            # Cartesian gym environment
│   └── lift_cube_cartesian.py  # Environment with Cartesian action space
└── robot/
    ├── real_robot.py           # SO-101 hardware interface
    └── joint_calibration.py    # Real→sim joint angle mapping

calibration/                    # Calibration scripts
├── calibrate_joints_real.py    # Joint mapping calibration (Step 2)
└── calibrate_3d.py             # Camera perspective correction calibration

.calibration/                   # Generated calibration data (gitignored)
└── joint_calibration.json      # Real robot joint angle offsets

# Key scripts (root)
run_real_ik.py                  # Real-robot IK controller runner
visualize_irl_block.py          # Block detection test / visualization
tests/test_aruco_homography_3d.py  # Homography calibration tool
make_aruco_board.py             # Generate printable calibration board (Step 1)
make_jenga_tag.py               # Generate printable Jenga block tag
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
