/**
 * detection.ts — ArUco tag detection using opencv-wasm.
 *
 * opencv-wasm is an optional peer dependency:
 *   npm install opencv-wasm
 *
 * @example
 * ```ts
 * import { Tag, Detector } from "lerobot-tagged";
 *
 * const detector = new Detector([
 *   new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 }),
 * ]);
 *
 * await detector.calibrate(frame);   // point camera at board, run once per session
 *
 * const locations = await detector.find(frame);
 * for (const loc of locations) {
 *   console.log(loc.tagId, loc.pos);
 * }
 * ```
 */

import { BoardConfig } from "./board.js";

/** Pixel coordinate [x, y]. */
export type PixelPoint = [number, number];

/** Result of a single tag detection. */
export interface TagLocation {
  /** ArUco tag ID. */
  tagId: number;
  /** Four corner pixels [TL, TR, BR, BL]. */
  cornersPx: [PixelPoint, PixelPoint, PixelPoint, PixelPoint];
  /** Mean of four corners. */
  centerPx: PixelPoint;
  /** Top-left corner pixel (primary reference point). */
  tlPx: PixelPoint;
  /** Position in board mm (origin = board top-left corner). */
  interiorMm: [number, number];
  /** Position after applying the tag's offset correction. */
  pos: [number, number];
}

/** Options passed to the {@link Tag} constructor. */
export interface TagOptions {
  /** [deltaX, deltaY] in mm added to the raw position. Defaults to [0, 0]. */
  offset?: [number, number];
  /** Physical width of the tracked object in mm. */
  widthMm?: number;
  /** Physical height of the tracked object in mm. */
  heightMm?: number;
  /** Physical depth of the tracked object in mm. */
  depthMm?: number;
  /** Path to an STL mesh for this object. Reserved for future use. */
  stlPath?: string;
}

/** A tag to detect, with optional pose correction and object dimensions. */
export class Tag {
  readonly id: number;
  readonly offset: [number, number];
  readonly widthMm?: number;
  readonly heightMm?: number;
  readonly depthMm?: number;
  readonly stlPath?: string;

  constructor(id: number, options: TagOptions = {}) {
    this.id = id;
    this.offset = options.offset ?? [0, 0];
    this.widthMm = options.widthMm;
    this.heightMm = options.heightMm;
    this.depthMm = options.depthMm;
    this.stlPath = options.stlPath;
  }
}

/**
 * Return the top-left corner position (in board mm) of every border tag,
 * mirroring the Python `border_tag_tl_positions()` function.
 */
function borderTagTlPositions(cfg: BoardConfig): Map<number, [number, number]> {
  const positions = new Map<number, [number, number]>();
  let tagId = 0;
  const half = cfg.tagMm / 2;

  for (const [inset, n] of [
    [0, cfg.nOuter],
    [cfg.tagMm + cfg.gapMm, cfg.nInner],
  ] as [number, number][]) {
    const margin = inset + half;
    const span = cfg.squareMm - 2 * margin;

    for (const side of ["top", "bottom", "left", "right"] as const) {
      for (let i = 0; i < n; i++) {
        const t = n > 1 ? i / (n - 1) : 0.5;
        const along = margin + t * span;
        let cx: number, cy: number;

        if (side === "top") {
          [cx, cy] = [along, margin];
        } else if (side === "bottom") {
          [cx, cy] = [along, cfg.squareMm - margin];
        } else if (side === "left") {
          if (i === 0 || i === n - 1) continue;
          [cx, cy] = [margin, along];
        } else {
          if (i === 0 || i === n - 1) continue;
          [cx, cy] = [cfg.squareMm - margin, along];
        }

        positions.set(tagId, [cx - half, cy - half]);
        tagId++;
      }
    }
  }
  return positions;
}

/** Apply a 3×3 homography (row-major, length 9) to a pixel point. */
function applyHomography(H: number[], px: number, py: number): [number, number] {
  const w = H[6] * px + H[7] * py + H[8];
  return [(H[0] * px + H[1] * py + H[2]) / w, (H[3] * px + H[4] * py + H[5]) / w];
}

async function loadCV(): Promise<any> {
  try {
    // @ts-ignore — optional peer dep
    const mod = await import("opencv-wasm");
    let cv = mod.default ?? mod;
    if (typeof cv.then === "function") cv = await cv;
    return cv;
  } catch {
    throw new Error(
      "opencv-wasm is required for tag detection. Install it with: npm install opencv-wasm"
    );
  }
}

/**
 * Detects a fixed set of ArUco tags (and in future, robot arm joints) in
 * camera frames.
 *
 * Call {@link calibrate} once with a frame that shows the full board, then
 * call {@link find} on every subsequent frame.
 *
 * @example
 * ```ts
 * const detector = new Detector([
 *   new Tag(101, { offset: [-5, 1], widthMm: 25, heightMm: 75, depthMm: 25 }),
 * ]);
 * await detector.calibrate(frame);
 * const locations = await detector.find(frame);
 * ```
 */
export class Detector {
  readonly tags: Map<number, Tag>;
  private _H: number[] | null = null;

  constructor(tags: Tag[]) {
    this.tags = new Map(tags.map(t => [t.id, t]));
  }

  /**
   * Compute the homography from a frame showing the full calibration board.
   *
   * Detects all visible border tags, matches them to their known positions,
   * and stores a homography mapping image pixels → board mm (origin = board
   * top-left corner).
   *
   * @param frame - ImageData with the full board visible.
   * @param cfg - Board configuration. Uses defaults if not provided.
   * @throws If fewer than 4 border tags are detected.
   */
  async calibrate(frame: ImageData, cfg: BoardConfig = new BoardConfig()): Promise<void> {
    const cv = await loadCV();
    const known = borderTagTlPositions(cfg);

    const src = cv.matFromImageData(frame);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const dict = cv.getPredefinedDictionary(cv.aruco.DICT_4X4_250);
    const params = new cv.aruco_DetectorParameters();
    const detector = new cv.aruco_ArucoDetector(dict, params);
    const corners = new cv.MatVector();
    const ids = new cv.Mat();
    const rejected = new cv.MatVector();
    detector.detectMarkers(gray, corners, ids, rejected);

    const srcPts: number[] = [];
    const dstPts: number[] = [];

    for (let i = 0; i < ids.rows; i++) {
      const tid = ids.intAt(i, 0);
      const knownPos = known.get(tid);
      if (!knownPos) continue;
      const cm = corners.get(i);
      srcPts.push(cm.floatAt(0, 0), cm.floatAt(0, 1)); // TL corner x, y
      dstPts.push(knownPos[0], knownPos[1]);
      cm.delete();
    }

    src.delete(); gray.delete(); corners.delete(); ids.delete();
    rejected.delete(); dict.delete(); params.delete(); detector.delete();

    if (srcPts.length < 8) { // 4 points × 2 coords
      throw new Error(
        `Only ${srcPts.length / 2} border tag(s) matched — need at least 4. ` +
        "Ensure the board is fully visible and well-lit."
      );
    }

    // Compute homography via least squares (DLT)
    // For a proper RANSAC implementation use opencv-wasm findHomography if available
    this._H = this._computeHomography(srcPts, dstPts);
  }

  /**
   * Detect all registered tags in a camera frame.
   *
   * @param frame - ImageData from a canvas or video element.
   * @returns Array of TagLocation. Tags not visible are omitted.
   * @throws If {@link calibrate} has not been called yet.
   */
  async find(frame: ImageData): Promise<TagLocation[]> {
    if (!this._H) throw new Error("Call calibrate() before find().");

    const cv = await loadCV();

    const src = cv.matFromImageData(frame);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const dict = cv.getPredefinedDictionary(cv.aruco.DICT_4X4_250);
    const params = new cv.aruco_DetectorParameters();
    const detector = new cv.aruco_ArucoDetector(dict, params);
    const corners = new cv.MatVector();
    const ids = new cv.Mat();
    const rejected = new cv.MatVector();
    detector.detectMarkers(gray, corners, ids, rejected);

    const results: TagLocation[] = [];
    const H = this._H;

    for (let i = 0; i < ids.rows; i++) {
      const tid = ids.intAt(i, 0);
      const tag = this.tags.get(tid);
      if (!tag) continue;

      const cm = corners.get(i);
      const pts: PixelPoint[] = Array.from({ length: 4 }, (_, j) => [
        cm.floatAt(0, j * 2),
        cm.floatAt(0, j * 2 + 1),
      ] as PixelPoint);
      const [tl, tr, br, bl] = pts as [PixelPoint, PixelPoint, PixelPoint, PixelPoint];
      const cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4;
      const cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4;

      const interiorMm = applyHomography(H, tl[0], tl[1]);
      results.push({
        tagId: tid,
        cornersPx: [tl, tr, br, bl],
        centerPx: [cx, cy],
        tlPx: tl,
        interiorMm,
        pos: [
          interiorMm[0] + tag.offset[0],
          interiorMm[1] + tag.offset[1],
        ],
      });
      cm.delete();
    }

    src.delete(); gray.delete(); corners.delete(); ids.delete();
    rejected.delete(); dict.delete(); params.delete(); detector.delete();

    return results;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /** Compute a 3×3 homography (row-major) from flat src/dst point arrays. */
  private _computeHomography(srcPts: number[], dstPts: number[]): number[] {
    const n = srcPts.length / 2;
    // Build the A matrix for the DLT algorithm (2n × 9)
    const A: number[][] = [];
    for (let i = 0; i < n; i++) {
      const [sx, sy] = [srcPts[i * 2], srcPts[i * 2 + 1]];
      const [dx, dy] = [dstPts[i * 2], dstPts[i * 2 + 1]];
      A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy, -dx]);
      A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy, -dy]);
    }
    // Solve via SVD — use the last row of V^T
    // Minimal SVD implementation for 9 unknowns
    const h = this._solveDLT(A);
    return h;
  }

  private _solveDLT(A: number[][]): number[] {
    // AtA
    const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
    for (const row of A) {
      for (let i = 0; i < 9; i++)
        for (let j = 0; j < 9; j++)
          AtA[i][j] += row[i] * row[j];
    }
    // Power iteration to find smallest eigenvector (approximation)
    // For production, replace with a proper SVD library
    let v = Array.from({ length: 9 }, (_, i) => i === 8 ? 1 : 0);
    const shift = AtA[8][8] + 1; // shift to target smallest eigenvalue
    const shifted = AtA.map((row, i) => row.map((val, j) => val - shift * (i === j ? 1 : 0)));
    // 200 iterations of inverse power method approximation
    for (let iter = 0; iter < 200; iter++) {
      const Av = shifted.map(row => row.reduce((s, val, j) => s + val * v[j], 0));
      const norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0));
      v = Av.map(x => x / (norm || 1));
    }
    // Normalise so h[8] = 1
    const scale = v[8] || 1;
    return v.map(x => x / scale);
  }
}
