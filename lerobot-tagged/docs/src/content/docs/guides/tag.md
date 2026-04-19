---
title: Tag Generation
description: Generate bordered ArUco tags for printing and attaching to objects.
sidebar:
  order: 2
---

A **bordered ArUco tag** has a thick outer border on all four sides and an L-shaped double-border indicator at the top-left corner. The L uniquely identifies orientation — no guessing which way is up.

```
┌─────────────────────┐
│ ┌──                 │  ← outer border (all sides)
│ │  ┌──────────────  │  ← L-indicator (top + left only)
│ │  │                │
│ │  │   ArUco bits   │
│ │  │                │
│ │  └──────────────  │
│ └──                 │
└─────────────────────┘
```

Print the tag, cut it out, and stick it to your object. The pick-101 pipeline uses **ID 101** on the Jenga block.

![Bordered ArUco tag ID 101](/pick-101/tag-101-example.png)

---

## Python

```python
from lerobot_tagged import generate_tag, save_tag_pdf, TagConfig
import cv2

# Save a print-ready PDF (A4, centred, 100% scale)
save_tag_pdf("tag_101.pdf", tag_id=101)

# Or get the image directly (grayscale ndarray)
img = generate_tag(tag_id=101)
cv2.imshow("tag", img)
cv2.waitKey(0)
```

### Custom size

```python
cfg = TagConfig()
cfg.tag_size_mm = 25    # physical ArUco grid width when printed (mm)
cfg.tag_size_px = 400   # render resolution

save_tag_pdf("tag_small.pdf", tag_id=42, cfg=cfg)
```

### TagConfig fields

| Field | Default | Description |
|---|---|---|
| `tag_size_px` | 400 | ArUco bit grid render size (px) |
| `quiet_zone_px` | 24 | White space between border and bits (px) |
| `border_px` | 10 | Outer border thickness (px) |
| `double_gap_px` | 8 | Gap between outer border and L-indicator (px) |
| `double_border_px` | 6 | L-indicator line thickness (px) |
| `double_length_ratio` | 0.45 | Fraction of side length covered by each indicator arm |
| `tag_size_mm` | 40 | Physical ArUco grid width when printed (mm) |

---

## JavaScript / TypeScript

```ts
import { saveTagPdf, TagConfig } from "lerobot-tagged";
import { writeFileSync } from "fs";

// Save a print-ready PDF
const pdfBytes = await saveTagPdf(101);
writeFileSync("tag_101.pdf", pdfBytes);

// Custom size
const cfg = new TagConfig();
cfg.tagSizeMm = 25;
const pdfBytes = await saveTagPdf(42, cfg);
```

### In the browser

```ts
import { saveTagPdf } from "lerobot-tagged";

const pdfBytes = await saveTagPdf(101);
const blob = new Blob([pdfBytes], { type: "application/pdf" });
window.open(URL.createObjectURL(blob));
```

### Preview on a canvas (no PDF)

```ts
import { drawTagToCanvas } from "lerobot-tagged";

const canvas = drawTagToCanvas(101);
document.body.appendChild(canvas);
```

:::note[Pixel-perfect ArUco bits]
`drawTagToCanvas` and `saveTagPdf` draw the border and L-indicator precisely, but render the ArUco bit area as a placeholder. For a fully accurate bit pattern, generate the ArUco image server-side with OpenCV and pass the PNG bytes as `renderedTagPng`:

```ts
// server.ts (Node + opencv-wasm or a Python endpoint)
const renderedPng: Uint8Array = /* OpenCV-generated ArUco PNG */;
const pdfBytes = await saveTagPdf(101, new TagConfig(), renderedPng);
```
:::

---

## Printing tips

- Print at **100% scale** — disable "fit to page" in your printer dialog.
- Use **plain white paper** — glossy surfaces can cause detection failures.
- After printing, verify the physical size of the ArUco grid with a ruler. It should match `tag_size_mm`.
