---
editUrl: false
next: false
prev: false
title: "saveTagPdf"
---

> **saveTagPdf**(`tagId`, `cfg?`, `renderedTagPng?`): `Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

Defined in: tag.ts:137

Generate a bordered ArUco tag and return it as a print-ready PDF (A4, centred).

Print at 100% scale — no page scaling.

## Parameters

### tagId

`number`

ArUco tag ID (0–249).

### cfg?

[`TagConfig`](/pick-101/api/classes/tagconfig/) = `...`

Rendering configuration.

### renderedTagPng?

`Uint8Array`\<`ArrayBufferLike`\>

Optional pre-rendered PNG of the ArUco bit grid
  (from OpenCV server-side). When provided, it is embedded as the tag image.

## Returns

`Promise`\<`Uint8Array`\<`ArrayBufferLike`\>\>

PDF bytes as Uint8Array.
