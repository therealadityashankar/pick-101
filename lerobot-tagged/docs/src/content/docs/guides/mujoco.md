---
title: MuJoCo Visualisation
description: Place detected objects in a MuJoCo scene and render it.
sidebar:
  order: 3
---

`TaggedScene` wraps a MuJoCo model so you can move a body to a detected tag position and render the result.

## Python

Install the optional `mujoco` dependency:

```bash
pip install "lerobot-tagged[mujoco]"
```

```python
from lerobot_tagged import TaggedScene

scene = TaggedScene.from_xml_path("models/so101/lift_cube.xml")

# Move the cube to the detected position
scene.set_object_position("cube", x=0.15, y=0.0, z=0.015)

# Render from a named camera → (H, W, 3) uint8 RGB array
rgb = scene.render(camera="topdown", width=600, height=600)

import cv2
cv2.imshow("sim", rgb[:, :, ::-1])
cv2.waitKey(0)
```

## JavaScript / TypeScript (browser + Node via WASM)

MuJoCo WASM must be loaded separately — see the [official WASM build](https://github.com/google-deepmind/mujoco/tree/main/wasm).

```ts
import { TaggedScene } from "lerobot-tagged";

// mj: initialised MuJoCo WASM module
const scene = TaggedScene.fromXmlString(mj, xmlString);

scene.setObjectPosition("cube", 0.15, 0.0, 0.015);

// Returns Uint8Array — width × height × 3, RGB row-major
const rgb = scene.render("topdown", 600, 600);
```

## API summary

### Python

| Method | Description |
|---|---|
| `TaggedScene(model, data=None)` | Construct from an `MjModel` |
| `TaggedScene.from_xml_path(path)` | Load from XML file |
| `TaggedScene.from_xml_string(xml)` | Load from XML string |
| `set_object_position(name, x, y, z, quat=…)` | Move a free-joint body |
| `render(camera, width, height)` | Render → `(H, W, 3)` ndarray |

### JavaScript

| Method | Description |
|---|---|
| `new TaggedScene(mj, model, data?)` | Construct from WASM module + model |
| `TaggedScene.fromXmlPath(mj, path)` | Load from file (Node only) |
| `TaggedScene.fromXmlString(mj, xml)` | Load from XML string |
| `setObjectPosition(name, x, y, z, quat?)` | Move a free-joint body |
| `render(camera, width, height)` | Render → `Uint8Array` (RGB) |
