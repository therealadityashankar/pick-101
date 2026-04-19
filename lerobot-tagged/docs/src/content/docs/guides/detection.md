---
title: Tag Detection
description: Detect ArUco tags in a camera frame and convert to board coordinates.
sidebar:
  order: 3
---

Point the camera at the board, call `detector.calibrate(frame)` once to compute
the homography, then call `detector.find(frame)` on every subsequent frame.
No files to save or load.

## Coordinate system

- **Origin**: board top-left corner.
- **X**: increases to the right (mm).
- **Y**: increases downward (mm).
- **Reference point**: tag top-left corner — `corners[0]` in OpenCV convention.

---

## Python

```python
import cv2
from lerobot_tagged import Tag, Detector

detector = Detector([
    Tag(101, offset=(-5.0, 1.0), width_mm=25, height_mm=75, depth_mm=25),
    Tag(42),
])

cap = cv2.VideoCapture(0)

# Calibrate — point camera at board, run once per session
_, frame = cap.read()
detector.calibrate(frame)

# Detect every frame
while True:
    _, frame = cap.read()

    for loc in detector.find(frame):
        x, y = loc.pos
        print(f"Tag {loc.tag_id} at ({x:.1f}, {y:.1f}) mm")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
```

Custom board config (if you printed with non-default settings):

```python
from lerobot_tagged import BoardConfig

cfg = BoardConfig()
cfg.square_mm = 180
cfg.tag_mm = 16
detector.calibrate(frame, cfg=cfg)
```

---

## JavaScript / TypeScript

### Install dependencies

```bash
npm install lerobot-tagged opencv-wasm
```

### Pre-warm opencv-wasm (optional but recommended)

opencv-wasm loads a WASM binary asynchronously. Pre-warming it on startup
means the first `calibrate()` call won't stall:

```ts
// opencv.ts
import cv2 from "opencv-wasm";

export const cvReady = new Promise<void>((resolve) => {
  cv2.onRuntimeInitialized = () => resolve();
});
```

### Detect

```ts
import { Tag, Detector } from "lerobot-tagged";
import { cvReady } from "./opencv.js";

await cvReady; // wait for WASM to load

const detector = new Detector([
  new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 }),
  new Tag(42),
]);

const video = document.querySelector("video")!;
await navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
  video.srcObject = s;
  return video.play();
});

const canvas = document.createElement("canvas");
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
const ctx = canvas.getContext("2d")!;

// Calibrate — point camera at board, run once per session
ctx.drawImage(video, 0, 0);
await detector.calibrate(ctx.getImageData(0, 0, canvas.width, canvas.height));

// Detect every frame
async function detect() {
  ctx.drawImage(video, 0, 0);
  const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (const loc of await detector.find(frame)) {
    const [x, y] = loc.pos;
    console.log(`Tag ${loc.tagId} at (${x.toFixed(1)}, ${y.toFixed(1)}) mm`);
  }

  requestAnimationFrame(detect);
}
detect();
```

---

## TagLocation fields

| Python | JS | Description |
|---|---|---|
| `tag_id` | `tagId` | ArUco tag ID |
| `corners_px` | `cornersPx` | Four corner pixels `[TL, TR, BR, BL]` |
| `center_px` | `centerPx` | Mean of four corners |
| `tl_px` | `tlPx` | Top-left corner pixel (primary reference) |
| `interior_mm` | `interiorMm` | Raw position in board mm |
| `pos` | `pos` | Position after offset correction |

---

## Choosing the offset

The `offset` on each `Tag` absorbs residual perspective error at the object's height:

1. Place the object at a known position on the board.
2. Read the raw `interior_mm` value from `detector.find()`.
3. `offset = known_position − raw_position`.
4. Verify across 4–8 positions — if consistent, you're done.

:::note[Planar assumption]
For tall objects the error grows and a fixed offset won't suffice — in that case
recalibrate with the camera at the object's height.
:::
