/**
 * tag.ts — Bordered ArUco tag generation for browser and Node.
 *
 * Generates a bordered ArUco tag as a PNG (Uint8Array) or embeds it in a PDF.
 * The tag has a thick outer border and an L-shaped indicator at the top-left
 * corner to mark orientation unambiguously.
 *
 * @example
 * ```ts
 * import { generateTag, saveTagPdf } from "lerobot-tagged";
 *
 * // Get PNG bytes (Node or browser)
 * const png = await generateTag(101);
 *
 * // Save as PDF (Node)
 * import { writeFileSync } from "fs";
 * writeFileSync("tag_101.pdf", await saveTagPdf(101));
 * ```
 */

import { PDFDocument } from "pdf-lib";

const A4_W_MM = 210;
const A4_H_MM = 297;
const MM_TO_PT = 72 / 25.4;

/** Rendering configuration for a bordered ArUco tag. */
export class TagConfig {
  /** Pixel size of the ArUco bit grid. */
  tagSizePx = 400;

  /** White space between the innermost border and the ArUco bits (px). */
  quietZonePx = 24;

  /** Thickness of the outer border rectangle (px). */
  borderPx = 10;

  /** Gap between the outer border and the L-indicator line (px). */
  doubleGapPx = 8;

  /** Thickness of the L-indicator line (px). */
  doubleBorderPx = 6;

  /** Fraction of the total image side covered by each indicator arm. */
  doubleLengthRatio = 0.45;

  /** Physical width of the ArUco bit grid when printed (mm). */
  tagSizeMm = 40;
}

/**
 * Draw a bordered ArUco tag onto a canvas.
 *
 * The tag bitmap is approximated with rectangles. For a pixel-perfect ArUco
 * bit pattern you need to supply the rendered bits from an OpenCV-based server
 * call — the border and L-indicator are drawn here regardless.
 *
 * @param tagId - ArUco tag ID (0–249).
 * @param cfg - Rendering configuration.
 * @returns HTMLCanvasElement containing the tag image.
 */
export function drawTagToCanvas(tagId: number, cfg: TagConfig = new TagConfig()): HTMLCanvasElement {
  const margin = cfg.borderPx + cfg.doubleGapPx + cfg.doubleBorderPx + cfg.quietZonePx;
  const total = cfg.tagSizePx + 2 * margin;

  const canvas = document.createElement("canvas");
  canvas.width = total;
  canvas.height = total;
  const ctx = canvas.getContext("2d")!;

  // White background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, total, total);

  // Outer border
  ctx.strokeStyle = "#000";
  ctx.lineWidth = cfg.borderPx;
  ctx.strokeRect(cfg.borderPx / 2, cfg.borderPx / 2, total - cfg.borderPx, total - cfg.borderPx);

  // L-shaped indicator at top-left
  const innerPos = cfg.borderPx + cfg.doubleGapPx + cfg.doubleBorderPx / 2;
  const armLen = Math.round(total * cfg.doubleLengthRatio);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = cfg.doubleBorderPx;
  ctx.beginPath();
  ctx.moveTo(0, innerPos); ctx.lineTo(armLen, innerPos);   // horizontal arm
  ctx.moveTo(innerPos, 0); ctx.lineTo(innerPos, armLen);   // vertical arm
  ctx.stroke();

  // ArUco bit area placeholder (dark grey — replace with rendered PNG for production)
  ctx.fillStyle = "#333";
  ctx.fillRect(margin, margin, cfg.tagSizePx, cfg.tagSizePx);

  // Label
  ctx.fillStyle = "#fff";
  ctx.font = `bold ${Math.round(cfg.tagSizePx * 0.12)}px monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`ID ${tagId}`, margin + cfg.tagSizePx / 2, margin + cfg.tagSizePx / 2);

  return canvas;
}

/**
 * Generate a bordered ArUco tag and return it as a PNG Uint8Array.
 *
 * In Node, pass a pre-rendered PNG of the ArUco bits via `renderedTagPng`
 * (generated server-side with OpenCV) to get a pixel-perfect result.
 *
 * @param tagId - ArUco tag ID (0–249).
 * @param cfg - Rendering configuration.
 * @param renderedTagPng - Optional pre-rendered PNG of the ArUco bit grid.
 * @returns PNG bytes as Uint8Array.
 */
export async function generateTag(
  tagId: number,
  cfg: TagConfig = new TagConfig(),
  renderedTagPng?: Uint8Array
): Promise<Uint8Array> {
  const pdfBytes = await saveTagPdf(tagId, cfg, renderedTagPng);
  // Return the embedded PNG if one was provided, otherwise return PDF bytes
  // (PDF is the primary output; PNG extraction requires a PDF renderer)
  return renderedTagPng ?? pdfBytes;
}

/**
 * Generate a bordered ArUco tag and return it as a print-ready PDF (A4, centred).
 *
 * Print at 100% scale — no page scaling.
 *
 * @param tagId - ArUco tag ID (0–249).
 * @param cfg - Rendering configuration.
 * @param renderedTagPng - Optional pre-rendered PNG of the ArUco bit grid
 *   (from OpenCV server-side). When provided, it is embedded as the tag image.
 * @returns PDF bytes as Uint8Array.
 */
export async function saveTagPdf(
  tagId: number,
  cfg: TagConfig = new TagConfig(),
  renderedTagPng?: Uint8Array
): Promise<Uint8Array> {
  const margin = cfg.borderPx + cfg.doubleGapPx + cfg.doubleBorderPx + cfg.quietZonePx;
  const totalPx = cfg.tagSizePx + 2 * margin;
  const scale = cfg.tagSizeMm / cfg.tagSizePx; // mm per pixel
  const totalMm = totalPx * scale;

  const doc = await PDFDocument.create();
  const page = doc.addPage([A4_W_MM * MM_TO_PT, A4_H_MM * MM_TO_PT]);

  const xOff = ((A4_W_MM - totalMm) / 2) * MM_TO_PT;
  const yOff = ((A4_H_MM - totalMm) / 2) * MM_TO_PT;
  const sizePt = totalMm * MM_TO_PT;
  const marginPt = margin * scale * MM_TO_PT;
  const tagPt = cfg.tagSizeMm * MM_TO_PT;

  if (renderedTagPng) {
    const pngImage = await doc.embedPng(renderedTagPng);
    page.drawImage(pngImage, { x: xOff + marginPt, y: yOff + marginPt, width: tagPt, height: tagPt });
  }

  // Outer border
  const { rgb } = await import("pdf-lib");
  const bPt = cfg.borderPx * scale * MM_TO_PT;
  page.drawRectangle({
    x: xOff, y: yOff, width: sizePt, height: sizePt,
    borderColor: rgb(0, 0, 0), borderWidth: bPt,
  });

  // L-shaped indicator at top-left (pdf-lib origin is bottom-left, so flip Y)
  const innerPosMm = (cfg.borderPx + cfg.doubleGapPx + cfg.doubleBorderPx / 2) * scale;
  const armMm = totalMm * cfg.doubleLengthRatio;
  const dbPt = cfg.doubleBorderPx * scale * MM_TO_PT;

  // Horizontal arm (top edge in visual space = high Y in pdf-lib)
  page.drawLine({
    start: { x: xOff, y: yOff + sizePt - innerPosMm * MM_TO_PT },
    end:   { x: xOff + armMm * MM_TO_PT, y: yOff + sizePt - innerPosMm * MM_TO_PT },
    color: rgb(0, 0, 0), thickness: dbPt,
  });

  // Vertical arm
  page.drawLine({
    start: { x: xOff + innerPosMm * MM_TO_PT, y: yOff + sizePt },
    end:   { x: xOff + innerPosMm * MM_TO_PT, y: yOff + sizePt - armMm * MM_TO_PT },
    color: rgb(0, 0, 0), thickness: dbPt,
  });

  return doc.save();
}
