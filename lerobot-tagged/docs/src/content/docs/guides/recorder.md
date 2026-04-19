---
title: Recording
description: Record tag detections over time and replay in MuJoCo.
sidebar:
  order: 4
---

`Recorder` wraps a camera and a `Detector`, captures frames in a background thread, and can replay the entire session as a MuJoCo video.

---

## Python

### Basic usage

```python
import cv2
import time
from lerobot_tagged import Tag, Detector, Recorder

detector = Detector([
    Tag(101, offset=(-5.0, 1.0), width_mm=25, height_mm=75, depth_mm=25),
])

cap = cv2.VideoCapture(0)

# Calibrate once before recording
_, frame = cap.read()
detector.calibrate(frame)

cap = cv2.VideoCapture(0)
recorder = Recorder(cap, detector)

recorder.record()        # starts background thread, non-blocking
time.sleep(10)           # record for 10 seconds
recorder.stop_recording()

print(f"Captured {len(recorder.frames)} frames")
for frame in recorder.frames[:3]:
    print(frame.timestamp, frame.tag_locations)
```

### MuJoCo replay

```python
# Auto-generate a scene from tag dimensions and write to mp4
recorder.output_mujoco_recording("output.mp4")

# Use a custom scene (your own MuJoCo XML)
recorder.output_mujoco_recording("output.mp4", xml_path="models/so101/lift_cube.xml")

# Custom camera and resolution
recorder.output_mujoco_recording(
    "output.mp4",
    camera="topdown",
    width=1280,
    height=720,
    fps=30.0,
)
```

When no `xml_path` is given, a scene is auto-generated with one box body per tracked tag, sized from `width_mm`, `height_mm`, `depth_mm`.

### Frame structure

```python
frame = recorder.frames[0]

frame.timestamp          # float — seconds since record() was called
frame.tag_locations      # List[TagLocation] — detected tags this frame
frame.joint_positions    # Dict[str, float] — joint angles (dummy zeros for now)
frame.end_effector       # Tuple[float, float, float] — end-effector xyz (dummy for now)
```

:::note[Joint positions]
`joint_positions` and `end_effector` currently return dummy zero values. A future update will wire these to real hardware readings so they appear in the MuJoCo replay alongside the object detections.
:::

---

## JavaScript / TypeScript

### Basic usage

```ts
import { Tag, Detector, Recorder } from "lerobot-tagged";

const detector = new Detector([
  new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 }),
]);

const video = document.querySelector("video")!;
await navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  return video.play();
});

// Calibrate once before recording
const canvas = document.createElement("canvas");
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
const ctx = canvas.getContext("2d")!;
ctx.drawImage(video, 0, 0);
await detector.calibrate(ctx.getImageData(0, 0, canvas.width, canvas.height));

const recorder = new Recorder(video, detector);

recorder.record();                           // non-blocking, uses requestAnimationFrame
await new Promise(r => setTimeout(r, 10000)); // record for 10 seconds
recorder.stopRecording();

console.log(`Captured ${recorder.frames.length} frames`);
```

### MuJoCo replay

```ts
// mj: initialised MuJoCo WASM module (see MuJoCo Visualisation guide)
const rgbFrames = await recorder.outputMujocoRecording(mj);
// rgbFrames: Uint8Array[] — one per frame, width × height × 3, RGB

// Custom scene XML
const xml = await fetch("/models/lift_cube.xml").then(r => r.text());
const rgbFrames = await recorder.outputMujocoRecording(mj, xml);

// In Node: frames are also written as PNGs to the output directory
// In browser: use rgbFrames with a canvas or MediaRecorder to encode video
```

### Frame structure

```ts
const frame = recorder.frames[0];

frame.timestamp        // number — seconds since record() was called
frame.tagLocations     // TagLocation[] — detected tags this frame
frame.jointPositions   // Record<string, number> — joint angles (dummy zeros for now)
frame.endEffector      // [number, number, number] — end-effector xyz (dummy for now)
```
