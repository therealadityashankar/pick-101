/**
 * recorder.ts — Record tag detections from a video source and replay in MuJoCo.
 *
 * @example
 * ```ts
 * import { Tag, Detector, Recorder } from "lerobot-tagged";
 *
 * const detector = new Detector(
 *   [new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 })],
 *   { H, scale, insetMm: 16 }
 * );
 *
 * const video = document.querySelector("video")!;
 * const recorder = new Recorder(video, detector);
 *
 * recorder.record();
 * // ... after some time ...
 * recorder.stopRecording();
 *
 * console.log(recorder.frames);
 * const renderedFrames = await recorder.outputMujocoRecording(mj);
 * ```
 */

import { Detector, TagLocation } from "./detection.js";

/** Dummy SO-101 joint names. Will be populated from hardware in a future update. */
const DUMMY_JOINTS: Record<string, number> = {
  shoulder_pan:  0,
  shoulder_lift: 0,
  elbow_flex:    0,
  wrist_flex:    0,
  wrist_roll:    0,
  gripper:       0,
};

/** A single recorded frame. */
export interface Frame {
  /** Seconds elapsed since {@link Recorder.record} was called. */
  timestamp: number;
  /** All detected tag locations in this frame. */
  tagLocations: TagLocation[];
  /**
   * Robot arm joint angles in radians, keyed by joint name.
   * Currently dummy zero values; will be populated from hardware in a future update.
   */
  jointPositions: Record<string, number>;
  /**
   * End-effector position in world metres [x, y, z].
   * Currently a dummy value; will be computed from joint positions in a future update.
   */
  endEffector: [number, number, number];
}

/**
 * Records tag detections from an HTMLVideoElement and replays them in MuJoCo.
 *
 * @example
 * ```ts
 * const recorder = new Recorder(videoElement, detector);
 * recorder.record();
 * await new Promise(r => setTimeout(r, 5000));
 * recorder.stopRecording();
 * const rgbFrames = await recorder.outputMujocoRecording(mj);
 * ```
 */
export class Recorder {
  private readonly video: HTMLVideoElement;
  private readonly detector: Detector;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  private _frames: Frame[] = [];
  private _rafHandle: number | null = null;
  private _startTime: number = 0;

  constructor(video: HTMLVideoElement, detector: Detector) {
    this.video = video;
    this.detector = detector;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d")!;
  }

  /** Start recording (non-blocking — runs on requestAnimationFrame). */
  record(): void {
    if (this._rafHandle !== null) {
      throw new Error("Recording is already in progress. Call stopRecording() first.");
    }
    this._frames = [];
    this._startTime = performance.now();
    this._loop();
  }

  /** Stop recording. Safe to call even if not currently recording. */
  stopRecording(): void {
    if (this._rafHandle !== null) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = null;
    }
  }

  /** All recorded frames in chronological order. */
  get frames(): Frame[] {
    return [...this._frames];
  }

  /**
   * Render all recorded frames in a MuJoCo scene and return the RGB frames.
   *
   * Each frame positions tracked objects according to their detected
   * ``pos`` coordinates (board interior mm → world metres:
   * X→X, Y→Y, both ÷ 1000; Z = half object height).
   *
   * In Node with the `fs` module available, frames are written as individual
   * PNGs to ``outputDir``. In both environments the rendered RGB Uint8Arrays
   * are returned for further processing.
   *
   * @param mj - Initialised MuJoCo WASM module.
   * @param xml - MuJoCo XML string. Defaults to an auto-generated scene with
   *   one box body per tracked tag.
   * @param outputDir - Directory to write PNG frames (Node only). Defaults to
   *   ``"mujoco_recording/"``
   * @param camera - MuJoCo camera name to render from.
   * @param width - Render width in pixels.
   * @param height - Render height in pixels.
   * @returns Array of rendered RGB frames (Uint8Array, width × height × 3).
   */
  async outputMujocoRecording(
    mj: any,
    xml?: string,
    outputDir: string = "mujoco_recording",
    camera: string = "topdown",
    width: number = 600,
    height: number = 600,
  ): Promise<Uint8Array[]> {
    if (this._frames.length === 0) {
      throw new Error("No frames recorded. Call record() and stopRecording() first.");
    }

    const sceneXml = xml ?? this._generateSceneXml();
    const model = mj.MjModel.from_xml_string(sceneXml);
    const data = new mj.MjData(model);
    mj.mj_forward(model, data);

    const renderer = new mj.Renderer(model, height, width);
    const rendered: Uint8Array[] = [];

    for (const frame of this._frames) {
      for (const loc of frame.tagLocations) {
        const tag = this.detector.tags.get(loc.tagId);
        const bodyName = `object_${loc.tagId}`;
        const bodyId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, bodyName);
        if (bodyId < 0) continue;

        const jntAdr = model.body_jntadr[bodyId];
        if (jntAdr < 0) continue;

        const [xMm, yMm] = loc.pos;
        const heightMm = tag?.heightMm ?? 25;
        const qposAdr = model.jnt_qposadr[jntAdr];
        data.qpos[qposAdr]     = xMm / 1000;
        data.qpos[qposAdr + 1] = yMm / 1000;
        data.qpos[qposAdr + 2] = heightMm / 2000;
        data.qpos[qposAdr + 3] = 1; // quaternion w
        data.qpos[qposAdr + 4] = 0;
        data.qpos[qposAdr + 5] = 0;
        data.qpos[qposAdr + 6] = 0;
      }

      mj.mj_forward(model, data);
      renderer.update_scene(data, camera);
      rendered.push(new Uint8Array(renderer.render()));
    }

    renderer.free();

    // Write to disk in Node
    try {
      const { writeFileSync, mkdirSync } = await import("fs");
      mkdirSync(outputDir, { recursive: true });
      rendered.forEach((frame, i) => {
        const padded = String(i).padStart(6, "0");
        writeFileSync(`${outputDir}/frame_${padded}.png`, frame);
      });
    } catch {
      // Not in Node — skip file writing
    }

    return rendered;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private _loop(): void {
    this._rafHandle = requestAnimationFrame(async () => {
      if (this._rafHandle === null) return; // stopped

      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      this.ctx.drawImage(this.video, 0, 0);
      const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

      const timestamp = (performance.now() - this._startTime) / 1000;
      const tagLocations = await this.detector.find(imageData);

      this._frames.push({
        timestamp,
        tagLocations,
        jointPositions: { ...DUMMY_JOINTS },
        endEffector: [0, 0, 0],
      });

      this._loop();
    });
  }

  private _generateSceneXml(): string {
    const bodies = [...this.detector.tags.values()].map(tag => {
      const w = (tag.widthMm  ?? 25) / 2000;
      const h = (tag.heightMm ?? 25) / 2000;
      const d = (tag.depthMm  ?? 25) / 2000;
      return `    <body name="object_${tag.id}" pos="0 0 ${h}">
      <freejoint/>
      <geom type="box" size="${w} ${d} ${h}" rgba="0.8 0.2 0.2 1"/>
    </body>`;
    }).join("\n");

    return `<mujoco model="lerobot-tagged-scene">
  <option timestep="0.002"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.85 0.85 0.85 1"/>
    <camera name="topdown" pos="0 0 0.8" euler="0 0 0"/>
${bodies}
  </worldbody>
</mujoco>`;
  }
}
