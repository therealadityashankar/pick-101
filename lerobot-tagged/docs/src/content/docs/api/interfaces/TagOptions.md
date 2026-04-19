---
editUrl: false
next: false
prev: false
title: "TagOptions"
---

Defined in: detection.ts:46

Options passed to the [Tag](/pick-101/api/classes/tag/) constructor.

## Properties

### depthMm?

> `optional` **depthMm?**: `number`

Defined in: detection.ts:54

Physical depth of the tracked object in mm.

***

### heightMm?

> `optional` **heightMm?**: `number`

Defined in: detection.ts:52

Physical height of the tracked object in mm.

***

### offset?

> `optional` **offset?**: \[`number`, `number`\]

Defined in: detection.ts:48

[deltaX, deltaY] in mm added to the raw position. Defaults to [0, 0].

***

### stlPath?

> `optional` **stlPath?**: `string`

Defined in: detection.ts:56

Path to an STL mesh for this object. Reserved for future use.

***

### widthMm?

> `optional` **widthMm?**: `number`

Defined in: detection.ts:50

Physical width of the tracked object in mm.
