---
editUrl: false
next: false
prev: false
title: "createArucoBoard"
---

> **createArucoBoard**(`cfg?`, `renderedBoardPng?`): `Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

Defined in: board.ts:125

Generate the ArUco calibration board as a PDF.

The ArUco tag bitmaps are drawn as black/white rectangles approximating
the marker pattern. For full-fidelity tag rendering, pass a rendered PNG
via the optional `renderedBoardPng` parameter (generated server-side with
OpenCV).

## Parameters

### cfg?

[`BoardConfig`](/pick-101/api/classes/boardconfig/) = `...`

Board configuration. Uses defaults if omitted.

### renderedBoardPng?

`Uint8Array`\<`ArrayBufferLike`\>

Optional pre-rendered board PNG as Uint8Array.
  When provided, this image is embedded directly instead of the simplified
  vector representation.

## Returns

`Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

PDF file as Uint8Array — write to disk or serve as a blob.
