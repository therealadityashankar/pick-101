---
editUrl: false
next: false
prev: false
title: "TagLocation"
---

Defined in: detection.ts:30

Result of a single tag detection.

## Properties

### centerPx

> **centerPx**: `PixelPoint`

Defined in: detection.ts:36

Mean of four corners.

***

### cornersPx

> **cornersPx**: \[`PixelPoint`, `PixelPoint`, `PixelPoint`, `PixelPoint`\]

Defined in: detection.ts:34

Four corner pixels [TL, TR, BR, BL].

***

### interiorMm

> **interiorMm**: \[`number`, `number`\]

Defined in: detection.ts:40

Position in board mm (origin = board top-left corner).

***

### pos

> **pos**: \[`number`, `number`\]

Defined in: detection.ts:42

Position after applying the tag's offset correction.

***

### tagId

> **tagId**: `number`

Defined in: detection.ts:32

ArUco tag ID.

***

### tlPx

> **tlPx**: `PixelPoint`

Defined in: detection.ts:38

Top-left corner pixel (primary reference point).
