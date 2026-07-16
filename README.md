# pick-101 - Aruco + IK controller

This repo lets one detect the location of jenga blocks and place jenga blocks appropriately via the use of inverse kinematics

some code was initially adapted from  forked from ggand0/pick-101 but, very little of that code still exists - except for the usage of dm_control, specifically the mapping within the src/ directory

## Installation

this package needs uv installed to be run properly, this can be downloaded from https://docs.astral.sh/uv/

```bash
git clone https://github.com/therealadityashankar/pick-101.git
cd pick-101
uv sync
```

---

## Running on the Real Robot

### Step 1, Print the board and block tag

```bash
uv run python tags-and-borders/make_aruco_board.py   # printables/aruco_board.pdf

uv run python tags-and-borders/make_jenga_tag.py     # printables/jenga_tag.pdf

# Full sheet of general-purpose bordered tags IDs 100–150
uv run python tags-and-borders/make_bordered_tags.py # printables/bordered_tags.pdf
```
NOTE : IMPORTANT do not  scale the pages while printing them - it needs to be printed at 100% scale, or the exact way it has been created

### Step 2, calibrate joints

Maps real robot joint readings to simulation joint angles. Produces `.calibration/joint_calibration.json`.

```bash
# full calibration
uv run python calibration/calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441

# single joint calibration
uv run python calibration/calibrate_joints_real.py --port /dev/tty.usbmodem5A680089441 --joint wrist_roll
```

### Step 3, Calibrate block position detection

Corrects for camera angle so the detected block position matches its true location on the board.

```bash
uv run python calibration/calibrate_board.py --camera 0
```

1. press C to start calibration
2. place the jenga block in the appropriately marked position
3. press space to set the position
4. repeat for all 4 corners
5. After 4 corners, the script fits a per-axis linear correction and saves `.calibration/camera_calibration.npz` — `run_real_ik.py` and `visualize_irl_block.py` load it automatically, no copy-pasting needed

### Step 4, Run on a real robot

```bash
uv run python run_real_ik.py --port /dev/tty.usbmodem5A680089441
```

Video is saved to `real_ik_run.mp4`.
