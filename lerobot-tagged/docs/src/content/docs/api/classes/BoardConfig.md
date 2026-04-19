---
editUrl: false
next: false
prev: false
title: "BoardConfig"
---

Defined in: board.ts:31

Physical configuration of the calibration board.

## Constructors

### Constructor

> **new BoardConfig**(): `BoardConfig`

#### Returns

`BoardConfig`

## Properties

### gapMm

> **gapMm**: `number` = `2`

Defined in: board.ts:39

Gap between inner and outer border tag rows in mm.

***

### nInner

> **nInner**: `number` = `8`

Defined in: board.ts:45

Number of tags along each inner border edge.

***

### nOuter

> **nOuter**: `number` = `10`

Defined in: board.ts:42

Number of tags along each outer border edge.

***

### robotTipLeftMm

> **robotTipLeftMm**: \[`number`, `number`\]

Defined in: board.ts:54

[x, y] position of the robot base LEFT tip in board mm.
Origin = board top-left corner.

***

### robotTipRadiusMm

> **robotTipRadiusMm**: `number` = `2`

Defined in: board.ts:63

Radius of the tip marker circles in mm.

***

### robotTipRightMm

> **robotTipRightMm**: \[`number`, `number`\]

Defined in: board.ts:60

[x, y] position of the robot base RIGHT tip in board mm.
Origin = board top-left corner.

***

### showRobotTips

> **showRobotTips**: `boolean` = `true`

Defined in: board.ts:48

Draw markers showing where the robot arm base tips should be placed.

***

### squareMm

> **squareMm**: `number` = `180`

Defined in: board.ts:33

Total board side length in mm.

***

### tagMm

> **tagMm**: `number` = `16`

Defined in: board.ts:36

Side length of each border ArUco tag in mm.
