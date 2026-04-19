/**
 * mujoco.ts — Tag-driven MuJoCo scene visualisation (browser + Node via WASM).
 *
 * Wraps a MuJoCo WASM model so that detected tag positions can be used to
 * place bodies (e.g. a cube) and render the scene.
 *
 * MuJoCo WASM must be loaded separately — see the official build at
 * https://github.com/google-deepmind/mujoco/tree/main/wasm
 *
 * @example
 * ```ts
 * import { TaggedScene } from "lerobot-tagged";
 *
 * // mj: MuJoCo WASM module (already initialised)
 * const scene = await TaggedScene.fromXmlPath(mj, "models/lift_cube.xml");
 * scene.setObjectPosition("cube", 0.15, 0.0, 0.015);
 * const rgb = scene.render("topdown", 600, 600);
 * ```
 */

/** Quaternion (w, x, y, z). */
export type Quat = [number, number, number, number];

/**
 * Thin wrapper around a MuJoCo WASM model for tag-driven visualisation.
 *
 * Use the static factory methods `fromXmlPath` or `fromXmlString` to create
 * instances rather than calling the constructor directly.
 */
export class TaggedScene {
  /** The raw MuJoCo WASM module. */
  readonly mj: any;
  /** The loaded MjModel. */
  readonly model: any;
  /** The MjData associated with the model. */
  readonly data: any;

  constructor(mj: any, model: any, data?: any) {
    this.mj = mj;
    this.model = model;
    this.data = data ?? new mj.MjData(model);
    mj.mj_forward(model, this.data);
  }

  /**
   * Load a model from an XML file path (Node / server-side only).
   *
   * @param mj - Initialised MuJoCo WASM module.
   * @param xmlPath - Path to the MuJoCo XML file.
   */
  static fromXmlPath(mj: any, xmlPath: string): TaggedScene {
    const model = mj.MjModel.from_xml_path(xmlPath);
    return new TaggedScene(mj, model);
  }

  /**
   * Load a model from an XML string (browser + Node).
   *
   * @param mj - Initialised MuJoCo WASM module.
   * @param xml - MuJoCo XML content as a string.
   */
  static fromXmlString(mj: any, xml: string): TaggedScene {
    const model = mj.MjModel.from_xml_string(xml);
    return new TaggedScene(mj, model);
  }

  /**
   * Move a free-joint body to the given world position and orientation.
   *
   * @param bodyName - Name of the MuJoCo body. Must have a free joint.
   * @param x - World X position in metres.
   * @param y - World Y position in metres.
   * @param z - World Z position in metres.
   * @param quat - Quaternion (w, x, y, z). Defaults to identity.
   */
  setObjectPosition(
    bodyName: string,
    x: number,
    y: number,
    z: number,
    quat: Quat = [1, 0, 0, 0]
  ): void {
    const { mj, model, data } = this;
    const bodyId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, bodyName);
    if (bodyId < 0) throw new Error(`Body '${bodyName}' not found in model.`);

    const jntAdr = model.body_jntadr[bodyId];
    if (jntAdr < 0) throw new Error(`Body '${bodyName}' has no joint.`);

    const qposAdr = model.jnt_qposadr[jntAdr];
    data.qpos[qposAdr]     = x;
    data.qpos[qposAdr + 1] = y;
    data.qpos[qposAdr + 2] = z;
    data.qpos[qposAdr + 3] = quat[0];
    data.qpos[qposAdr + 4] = quat[1];
    data.qpos[qposAdr + 5] = quat[2];
    data.qpos[qposAdr + 6] = quat[3];
    for (let i = 0; i < 6; i++) data.qvel[qposAdr + i] = 0;
    mj.mj_forward(model, data);
  }

  /**
   * Render the scene and return an RGB pixel buffer.
   *
   * @param camera - Name of the MuJoCo camera to render from.
   * @param width - Output image width in pixels.
   * @param height - Output image height in pixels.
   * @returns Uint8Array of length width × height × 3 (RGB, row-major).
   */
  render(
    camera: string = "topdown",
    width: number = 600,
    height: number = 600
  ): Uint8Array {
    const { mj, model, data } = this;
    const renderer = new mj.Renderer(model, height, width);
    renderer.update_scene(data, camera);
    const rgb = renderer.render();
    const copy = new Uint8Array(rgb);
    renderer.free();
    return copy;
  }
}
