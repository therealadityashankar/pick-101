---
title: Calibration Board
description: Generate and print the ArUco calibration board.
sidebar:
  order: 1
---

The calibration board is a square grid of ArUco tags arranged in two concentric border rings. Print it on A4 paper at **100% scale** — no page scaling.

## Python

```python
from lerobot_tagged import create_aruco_board_pdf, BoardConfig

# Default board (180 × 180 mm on A4)
create_aruco_board_pdf("board.pdf")

# Custom configuration
cfg = BoardConfig()
cfg.square_mm = 180
cfg.tag_mm = 16
cfg.gap_mm = 2
cfg.n_outer = 10
cfg.n_inner = 8
cfg.show_robot_tips = True
cfg.robot_tip_left_mm = (80, 180)
cfg.robot_tip_right_mm = (100, 180)
create_aruco_board_pdf("board.pdf", cfg)
```

## JavaScript / TypeScript

```ts
import { createArucoBoard, BoardConfig } from "lerobot-tagged";
import { writeFileSync } from "fs";

const cfg = new BoardConfig();
cfg.squareMm = 180;
cfg.showRobotTips = true;

const pdfBytes = await createArucoBoard(cfg);

// Node
writeFileSync("board.pdf", pdfBytes);

// Browser
const blob = new Blob([pdfBytes], { type: "application/pdf" });
window.open(URL.createObjectURL(blob));
```

## Configuration reference

| Python field | JS field | Default | Description |
|---|---|---|---|
| `square_mm` | `squareMm` | 180 | Board side length (mm) |
| `tag_mm` | `tagMm` | 16 | ArUco tag side length (mm) |
| `gap_mm` | `gapMm` | 2 | Gap between inner and outer rings (mm) |
| `n_outer` | `nOuter` | 10 | Tags per outer edge |
| `n_inner` | `nInner` | 8 | Tags per inner edge |
| `show_robot_tips` | `showRobotTips` | `True` | Draw robot base alignment markers |
| `robot_tip_left_mm` | `robotTipLeftMm` | `(80, 180)` | Left tip position in board mm |
| `robot_tip_right_mm` | `robotTipRightMm` | `(100, 180)` | Right tip position in board mm |

## Tag IDs

Tags are numbered sequentially from 0: outer ring first (top → bottom → left → right edges), then inner ring. The Jenga block tag uses **ID 101** from the 4×4\_250 dictionary.
