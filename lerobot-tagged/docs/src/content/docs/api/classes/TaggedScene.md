---
editUrl: false
next: false
prev: false
title: "TaggedScene"
---

Defined in: mujoco.ts:30

Thin wrapper around a MuJoCo WASM model for tag-driven visualisation.

Use the static factory methods `fromXmlPath` or `fromXmlString` to create
instances rather than calling the constructor directly.

## Constructors

### Constructor

> **new TaggedScene**(`mj`, `model`, `data?`): `TaggedScene`

Defined in: mujoco.ts:38

#### Parameters

##### mj

`any`

##### model

`any`

##### data?

`any`

#### Returns

`TaggedScene`

## Properties

### data

> `readonly` **data**: `any`

Defined in: mujoco.ts:36

The MjData associated with the model.

***

### mj

> `readonly` **mj**: `any`

Defined in: mujoco.ts:32

The raw MuJoCo WASM module.

***

### model

> `readonly` **model**: `any`

Defined in: mujoco.ts:34

The loaded MjModel.

## Methods

### render()

> **render**(`camera?`, `width?`, `height?`): `Uint8Array`

Defined in: mujoco.ts:110

Render the scene and return an RGB pixel buffer.

#### Parameters

##### camera?

`string` = `"topdown"`

Name of the MuJoCo camera to render from.

##### width?

`number` = `600`

Output image width in pixels.

##### height?

`number` = `600`

Output image height in pixels.

#### Returns

`Uint8Array`

Uint8Array of length width × height × 3 (RGB, row-major).

***

### setObjectPosition()

> **setObjectPosition**(`bodyName`, `x`, `y`, `z`, `quat?`): `void`

Defined in: mujoco.ts:76

Move a free-joint body to the given world position and orientation.

#### Parameters

##### bodyName

`string`

Name of the MuJoCo body. Must have a free joint.

##### x

`number`

World X position in metres.

##### y

`number`

World Y position in metres.

##### z

`number`

World Z position in metres.

##### quat?

`Quat` = `...`

Quaternion (w, x, y, z). Defaults to identity.

#### Returns

`void`

***

### fromXmlPath()

> `static` **fromXmlPath**(`mj`, `xmlPath`): `TaggedScene`

Defined in: mujoco.ts:51

Load a model from an XML file path (Node / server-side only).

#### Parameters

##### mj

`any`

Initialised MuJoCo WASM module.

##### xmlPath

`string`

Path to the MuJoCo XML file.

#### Returns

`TaggedScene`

***

### fromXmlString()

> `static` **fromXmlString**(`mj`, `xml`): `TaggedScene`

Defined in: mujoco.ts:62

Load a model from an XML string (browser + Node).

#### Parameters

##### mj

`any`

Initialised MuJoCo WASM module.

##### xml

`string`

MuJoCo XML content as a string.

#### Returns

`TaggedScene`
