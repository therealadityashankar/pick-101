---
editUrl: false
next: false
prev: false
title: "generateTag"
---

> **generateTag**(`tagId`, `cfg?`, `renderedTagPng?`): `Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

Defined in: tag.ts:115

Generate a bordered ArUco tag and return it as a PNG Uint8Array.

In Node, pass a pre-rendered PNG of the ArUco bits via `renderedTagPng`
(generated server-side with OpenCV) to get a pixel-perfect result.

## Parameters

### tagId

`number`

ArUco tag ID (0–249).

### cfg?

[`TagConfig`](/pick-101/api/classes/tagconfig/) = `...`

Rendering configuration.

### renderedTagPng?

`Uint8Array`\<`ArrayBufferLike`\>

Optional pre-rendered PNG of the ArUco bit grid.

## Returns

`Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

PNG bytes as Uint8Array.
