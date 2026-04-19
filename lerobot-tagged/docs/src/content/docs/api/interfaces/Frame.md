---
editUrl: false
next: false
prev: false
title: "Frame"
---

Defined in: recorder.ts:38

A single recorded frame.

## Properties

### endEffector

> **endEffector**: \[`number`, `number`, `number`\]

Defined in: recorder.ts:52

End-effector position in world metres [x, y, z].
Currently a dummy value; will be computed from joint positions in a future update.

***

### jointPositions

> **jointPositions**: `Record`\<`string`, `number`\>

Defined in: recorder.ts:47

Robot arm joint angles in radians, keyed by joint name.
Currently dummy zero values; will be populated from hardware in a future update.

***

### tagLocations

> **tagLocations**: [`TagLocation`](/pick-101/api/interfaces/taglocation/)[]

Defined in: recorder.ts:42

All detected tag locations in this frame.

***

### timestamp

> **timestamp**: `number`

Defined in: recorder.ts:40

Seconds elapsed since [Recorder.record](/pick-101/api/classes/recorder/#record) was called.
