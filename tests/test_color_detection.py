"""Capture a frame from a camera and test color detection for sim-to-real cube.

Usage:
    python test_color_detection.py --camera 0
    python test_color_detection.py --camera 1 --save result.png
"""
import argparse
import cv2
import numpy as np


# HSV ranges for each target color
# Hue is 0-179 in OpenCV
COLOR_RANGES = {
    "bright_blue": {
        "lower": np.array([90, 120, 80]),
        "upper": np.array([115, 255, 255]),
        "bgr": (255, 191, 0),  # for overlay
    },
}


def detect_colors(frame: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = {}

    for name, cfg in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        results[name] = {
            "mask": mask,
            "contours": contours,
            "largest_area": cv2.contourArea(contours[0]) if contours else 0,
            "bgr": cfg["bgr"],
        }

    return results


def find_quad_corners(contour: np.ndarray) -> np.ndarray | None:
    """Approximate contour to a quadrilateral and return its 4 corners."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)

    # Fall back: find the 4 extremal corners of the convex hull
    hull = cv2.convexHull(contour, returnPoints=True).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None

    # Top-left, top-right, bottom-right, bottom-left by sum/diff
    s = hull.sum(axis=1)
    d = np.diff(hull, axis=1).ravel()
    corners = np.array([
        hull[np.argmin(s)],
        hull[np.argmin(d)],
        hull[np.argmax(s)],
        hull[np.argmax(d)],
    ], dtype=np.int32)
    return corners


def draw_detections(frame: np.ndarray, results: dict) -> np.ndarray:
    out = frame.copy()
    for name, r in results.items():
        if not r["contours"]:
            continue
        c = r["contours"][0]
        if r["largest_area"] < 100:
            continue

        corners = find_quad_corners(c)
        if corners is not None:
            pts = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=r["bgr"], thickness=2)
            for pt in corners:
                cv2.circle(out, tuple(pt.astype(int)), 5, r["bgr"], -1)

        x, y, w, h = cv2.boundingRect(c)
        label = f"{name} ({r['largest_area']:.0f}px²)"
        cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, r["bgr"], 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--save", type=str, default=None, help="Save result image to this path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    print(f"Opened camera {args.camera}. Press SPACE to capture, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        results = detect_colors(frame)
        preview = draw_detections(frame, results)

        # Show masks side by side at bottom
        h, w = frame.shape[:2]
        mask_display = np.zeros((120, w, 3), dtype=np.uint8)
        names = list(results.keys())
        panel_w = w // len(names)
        for i, name in enumerate(names):
            mask_small = cv2.resize(results[name]["mask"], (panel_w - 4, 116))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_display[2:118, i * panel_w: i * panel_w + panel_w - 4] = mask_bgr
            cv2.putText(mask_display, name, (i * panel_w + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        combined = np.vstack([preview, mask_display])
        cv2.imshow("Color Detection", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            save_path = args.save or "color_detection_result.png"
            cv2.imwrite(save_path, combined)
            print(f"Saved to {save_path}")
            for name, r in results.items():
                print(f"  {name}: largest contour area = {r['largest_area']:.0f} px²")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
