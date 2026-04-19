---
editUrl: false
next: false
prev: false
title: "Detector"
---

Defined in: detection.ts:156

Detects a fixed set of ArUco tags (and in future, robot arm joints) in
camera frames.

Call [calibrate](/pick-101/api/classes/detector/#calibrate) once with a frame that shows the full board, then
call [find](/pick-101/api/classes/detector/#find) on every subsequent frame.

## Example

```ts
const detector = new Detector([
  new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 }),
]);
await detector.calibrate(frame);
const locations = await detector.find(frame);
```

## Constructors

### Constructor

> **new Detector**(`tags`): `Detector`

Defined in: detection.ts:160

#### Parameters

##### tags

[`Tag`](/pick-101/api/classes/tag/)[]

#### Returns

`Detector`

## Properties

### tags

> `readonly` **tags**: `Map`\<`number`, [`Tag`](/pick-101/api/classes/tag/)\>

Defined in: detection.ts:157

## Methods

### calibrate()

> **calibrate**(`frame`, `cfg?`): `Promise`\<`void`\>

Defined in: detection.ts:175

Compute the homography from a frame showing the full calibration board.

Detects all visible border tags, matches them to their known positions,
and stores a homography mapping image pixels → board mm (origin = board
top-left corner).

#### Parameters

##### frame

`ImageData`

ImageData with the full board visible.

##### cfg?

[`BoardConfig`](/pick-101/api/classes/boardconfig/) = `...`

Board configuration. Uses defaults if not provided.

#### Returns

`Promise`\<`void`\>

#### Throws

If fewer than 4 border tags are detected.

***

### find()

> **find**(`frame`): `Promise`\<[`TagLocation`](/pick-101/api/interfaces/taglocation/)[]\>

Defined in: detection.ts:226

Detect all registered tags in a camera frame.

#### Parameters

##### frame

`ImageData`

ImageData from a canvas or video element.

#### Returns

`Promise`\<[`TagLocation`](/pick-101/api/interfaces/taglocation/)[]\>

Array of TagLocation. Tags not visible are omitted.

#### Throws

If [calibrate](/pick-101/api/classes/detector/#calibrate) has not been called yet.
