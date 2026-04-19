---
editUrl: false
next: false
prev: false
title: "drawTagToCanvas"
---

> **drawTagToCanvas**(`tagId`, `cfg?`): `HTMLCanvasElement`

Defined in: tag.ts:62

Draw a bordered ArUco tag onto a canvas.

The tag bitmap is approximated with rectangles. For a pixel-perfect ArUco
bit pattern you need to supply the rendered bits from an OpenCV-based server
call — the border and L-indicator are drawn here regardless.

## Parameters

### tagId

`number`

ArUco tag ID (0–249).

### cfg?

[`TagConfig`](/pick-101/api/classes/tagconfig/) = `...`

Rendering configuration.

## Returns

`HTMLCanvasElement`

HTMLCanvasElement containing the tag image.
