---
editUrl: false
next: false
prev: false
title: "Recorder"
---

Defined in: recorder.ts:67

Records tag detections from an HTMLVideoElement and replays them in MuJoCo.

## Example

```ts
const recorder = new Recorder(videoElement, detector);
recorder.record();
await new Promise(r => setTimeout(r, 5000));
recorder.stopRecording();
const rgbFrames = await recorder.outputMujocoRecording(mj);
```

## Constructors

### Constructor

> **new Recorder**(`video`, `detector`): `Recorder`

Defined in: recorder.ts:77

#### Parameters

##### video

`HTMLVideoElement`

##### detector

[`Detector`](/pick-101/api/classes/detector/)

#### Returns

`Recorder`

## Accessors

### frames

#### Get Signature

> **get** **frames**(): [`Frame`](/pick-101/api/interfaces/frame/)[]

Defined in: recorder.ts:103

All recorded frames in chronological order.

##### Returns

[`Frame`](/pick-101/api/interfaces/frame/)[]

## Methods

### outputMujocoRecording()

> **outputMujocoRecording**(`mj`, `xml?`, `outputDir?`, `camera?`, `width?`, `height?`): `Promise`\<`Uint8Array`\<`ArrayBufferLike`\>[]\>

Defined in: recorder.ts:128

Render all recorded frames in a MuJoCo scene and return the RGB frames.

Each frame positions tracked objects according to their detected
``pos`` coordinates (board interior mm → world metres:
X→X, Y→Y, both ÷ 1000; Z = half object height).

In Node with the `fs` module available, frames are written as individual
PNGs to ``outputDir``. In both environments the rendered RGB Uint8Arrays
are returned for further processing.

#### Parameters

##### mj

`any`

Initialised MuJoCo WASM module.

##### xml?

`string`

MuJoCo XML string. Defaults to an auto-generated scene with
  one box body per tracked tag.

##### outputDir?

`string` = `"mujoco_recording"`

Directory to write PNG frames (Node only). Defaults to
  ``"mujoco_recording/"``

##### camera?

`string` = `"topdown"`

MuJoCo camera name to render from.

##### width?

`number` = `600`

Render width in pixels.

##### height?

`number` = `600`

Render height in pixels.

#### Returns

`Promise`\<`Uint8Array`\<`ArrayBufferLike`\>[]\>

Array of rendered RGB frames (Uint8Array, width × height × 3).

***

### record()

> **record**(): `void`

Defined in: recorder.ts:85

Start recording (non-blocking — runs on requestAnimationFrame).

#### Returns

`void`

***

### stopRecording()

> **stopRecording**(): `void`

Defined in: recorder.ts:95

Stop recording. Safe to call even if not currently recording.

#### Returns

`void`
