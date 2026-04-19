/**
 * board.ts — ArUco calibration board generation for browser and Node.
 *
 * Generates the calibration board as a PDF using pdf-lib.
 * The board includes ArUco border tags and optional robot tip markers.
 *
 * @example
 * ```ts
 * import { createArucoBoard, BoardConfig } from "lerobot-tagged";
 *
 * const cfg = new BoardConfig();
 * const pdfBytes = await createArucoBoard(cfg);
 *
 * // In Node
 * import { writeFileSync } from "fs";
 * writeFileSync("board.pdf", pdfBytes);
 *
 * // In browser
 * const blob = new Blob([pdfBytes], { type: "application/pdf" });
 * const url = URL.createObjectURL(blob);
 * ```
 */

import { PDFDocument, rgb, degrees } from "pdf-lib";

const A4_W_MM = 210;
const A4_H_MM = 297;
const MM_TO_PT = 72 / 25.4; // 1 mm = 2.835 pt

/** Physical configuration of the calibration board. */
export class BoardConfig {
  /** Total board side length in mm. */
  squareMm = 180;

  /** Side length of each border ArUco tag in mm. */
  tagMm = 16;

  /** Gap between inner and outer border tag rows in mm. */
  gapMm = 2;

  /** Number of tags along each outer border edge. */
  nOuter = 10;

  /** Number of tags along each inner border edge. */
  nInner = 8;

  /** Draw markers showing where the robot arm base tips should be placed. */
  showRobotTips = true;

  /**
   * [x, y] position of the robot base LEFT tip in board mm.
   * Origin = board top-left corner.
   */
  robotTipLeftMm: [number, number] = [80, 180];

  /**
   * [x, y] position of the robot base RIGHT tip in board mm.
   * Origin = board top-left corner.
   */
  robotTipRightMm: [number, number] = [100, 180];

  /** Radius of the tip marker circles in mm. */
  robotTipRadiusMm = 2;
}

interface TagCentre {
  id: number;
  xMm: number;
  yMm: number;
}

function sideCentres(
  squareMm: number,
  tagMm: number,
  insetMm: number,
  n: number,
  side: "top" | "bottom" | "left" | "right"
): Array<[number, number]> {
  const halfMm = tagMm / 2;
  const margin = insetMm + halfMm;
  const span = squareMm - 2 * margin;
  const out: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0.5;
    const along = margin + t * span;
    if (side === "top") out.push([along, insetMm + halfMm]);
    else if (side === "bottom") out.push([along, squareMm - insetMm - halfMm]);
    else if (side === "left" && i > 0 && i < n - 1) out.push([insetMm + halfMm, along]);
    else if (side === "right" && i > 0 && i < n - 1)
      out.push([squareMm - insetMm - halfMm, along]);
  }
  return out;
}

function allTagCentres(cfg: BoardConfig): TagCentre[] {
  const centres: TagCentre[] = [];
  let id = 0;
  for (const [inset, n] of [
    [0, cfg.nOuter],
    [cfg.tagMm + cfg.gapMm, cfg.nInner],
  ] as [number, number][]) {
    for (const side of ["top", "bottom", "left", "right"] as const) {
      for (const [x, y] of sideCentres(cfg.squareMm, cfg.tagMm, inset, n, side)) {
        centres.push({ id: id++, xMm: x, yMm: y });
      }
    }
  }
  return centres;
}

/**
 * Generate the ArUco calibration board as a PDF.
 *
 * The ArUco tag bitmaps are drawn as black/white rectangles approximating
 * the marker pattern. For full-fidelity tag rendering, pass a rendered PNG
 * via the optional `renderedBoardPng` parameter (generated server-side with
 * OpenCV).
 *
 * @param cfg Board configuration. Uses defaults if omitted.
 * @param renderedBoardPng Optional pre-rendered board PNG as Uint8Array.
 *   When provided, this image is embedded directly instead of the simplified
 *   vector representation.
 * @returns PDF file as Uint8Array — write to disk or serve as a blob.
 */
export async function createArucoBoard(
  cfg: BoardConfig = new BoardConfig(),
  renderedBoardPng?: Uint8Array
): Promise<Uint8Array> {
  const doc = await PDFDocument.create();
  const page = doc.addPage([A4_W_MM * MM_TO_PT, A4_H_MM * MM_TO_PT]);

  const boardPt = cfg.squareMm * MM_TO_PT;
  const xOff = ((A4_W_MM - cfg.squareMm) / 2) * MM_TO_PT;
  // pdf-lib origin is bottom-left; board is centred vertically
  const yOff = ((A4_H_MM - cfg.squareMm) / 2) * MM_TO_PT;

  if (renderedBoardPng) {
    // Embed pre-rendered PNG (highest fidelity)
    const pngImage = await doc.embedPng(renderedBoardPng);
    page.drawImage(pngImage, { x: xOff, y: yOff, width: boardPt, height: boardPt });
  } else {
    // Simplified vector fallback: draw board outline + tag position markers
    page.drawRectangle({
      x: xOff,
      y: yOff,
      width: boardPt,
      height: boardPt,
      borderColor: rgb(0, 0, 0),
      borderWidth: 1,
    });

    const tagPt = cfg.tagMm * MM_TO_PT;
    for (const { xMm, yMm } of allTagCentres(cfg)) {
      // pdf-lib Y is bottom-up; flip within board
      const tx = xOff + (xMm - cfg.tagMm / 2) * MM_TO_PT;
      const ty = yOff + (cfg.squareMm - yMm - cfg.tagMm / 2) * MM_TO_PT;
      page.drawRectangle({
        x: tx,
        y: ty,
        width: tagPt,
        height: tagPt,
        color: rgb(0.2, 0.2, 0.2),
      });
    }
  }

  // Robot tip markers
  if (cfg.showRobotTips) {
    const rPt = cfg.robotTipRadiusMm * MM_TO_PT;
    for (const [xMm, yMm] of [cfg.robotTipLeftMm, cfg.robotTipRightMm]) {
      const tx = xOff + xMm * MM_TO_PT;
      const ty = yOff + (cfg.squareMm - yMm) * MM_TO_PT; // flip Y
      const armPt = rPt * 3;
      // Crosshair
      page.drawLine({
        start: { x: tx - armPt, y: ty },
        end:   { x: tx + armPt, y: ty },
        color: rgb(0.8, 0, 0),
        thickness: rPt / 2,
      });
      page.drawLine({
        start: { x: tx, y: ty - armPt },
        end:   { x: tx, y: ty + armPt },
        color: rgb(0.8, 0, 0),
        thickness: rPt / 2,
      });
      // Circle
      page.drawCircle({
        x: tx,
        y: ty,
        size: rPt,
        color: rgb(0.8, 0, 0),
      });
    }
  }

  // Scale bar (10 mm)
  const barX = xOff;
  const barY = yOff - 8 * MM_TO_PT;
  const barLen = 10 * MM_TO_PT;
  page.drawLine({ start: { x: barX, y: barY }, end: { x: barX + barLen, y: barY },
                  color: rgb(0, 0, 0), thickness: 0.5 });
  page.drawLine({ start: { x: barX, y: barY - 2 }, end: { x: barX, y: barY + 2 },
                  color: rgb(0, 0, 0), thickness: 0.5 });
  page.drawLine({ start: { x: barX + barLen, y: barY - 2 },
                  end:   { x: barX + barLen, y: barY + 2 },
                  color: rgb(0, 0, 0), thickness: 0.5 });

  return doc.save();
}
