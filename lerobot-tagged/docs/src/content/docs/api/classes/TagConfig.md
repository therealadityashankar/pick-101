---
editUrl: false
next: false
prev: false
title: "TagConfig"
---

Defined in: tag.ts:28

Rendering configuration for a bordered ArUco tag.

## Constructors

### Constructor

> **new TagConfig**(): `TagConfig`

#### Returns

`TagConfig`

## Properties

### borderPx

> **borderPx**: `number` = `10`

Defined in: tag.ts:36

Thickness of the outer border rectangle (px).

***

### doubleBorderPx

> **doubleBorderPx**: `number` = `6`

Defined in: tag.ts:42

Thickness of the L-indicator line (px).

***

### doubleGapPx

> **doubleGapPx**: `number` = `8`

Defined in: tag.ts:39

Gap between the outer border and the L-indicator line (px).

***

### doubleLengthRatio

> **doubleLengthRatio**: `number` = `0.45`

Defined in: tag.ts:45

Fraction of the total image side covered by each indicator arm.

***

### quietZonePx

> **quietZonePx**: `number` = `24`

Defined in: tag.ts:33

White space between the innermost border and the ArUco bits (px).

***

### tagSizeMm

> **tagSizeMm**: `number` = `40`

Defined in: tag.ts:48

Physical width of the ArUco bit grid when printed (mm).

***

### tagSizePx

> **tagSizePx**: `number` = `400`

Defined in: tag.ts:30

Pixel size of the ArUco bit grid.
