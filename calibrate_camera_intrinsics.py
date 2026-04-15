"""Intrinsic camera calibration using a checkerboard pattern.

Captures frames from the real wrist camera, detects checkerboard corners,
then runs cv2.calibrateCamera() to get the camera matrix and distortion
coefficients (radial + tangential distortion).

Saved to: calibration/camera_intrinsics.json

Usage:
    python calibrate_camera_intrinsics.py --camera 1
    python calibrate_camera_intrinsics.py --camera 1 --cols 9 --rows 6 --square 0.025

Controls:
    SPACE      — capture current frame (if corners found)
    C          — run calibration with captured frames so far
    U          — toggle undistort preview on/off
    Q / ESC    — quit (saves if calibration was run)

Tips:
    - Use a printed checkerboard (default: 9x6 inner corners, 25mm squares)
    - Capture 15-25 frames from different angles and distances
    - Make sure the board fills various parts of the frame
    - The green overlay shows detected corners — only captures when detected
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

SAVE_PATH = Path("calibration/camera_intrinsics.json")

# Checkerboard defaults (inner corners, not squares)
DEFAULT_COLS = 9
DEFAULT_ROWS = 6
DEFAULT_SQUARE_M = 0.030  # 30 mm squares


def calibrate(obj_points, img_points, img_shape):
    """Run cv2.calibrateCamera and return (camera_matrix, dist_coeffs, rms_error)."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )
    return K, dist, rms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Inner corners along width")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Inner corners along height")
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE_M, help="Square size in metres")
    args = parser.parse_args()

    SAVE_PATH.parent.mkdir(exist_ok=True)

    board_size = (args.cols, args.rows)
    square_m   = args.square

    # 3D object points for one board pose (Z=0 plane)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * square_m

    obj_points = []   # 3D points per captured frame
    img_points = []   # 2D points per captured frame

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    K, dist = None, None
    show_undistort = False
    img_shape = None

    print(f"Checkerboard: {args.cols}x{args.rows} inner corners, {square_m*1000:.0f}mm squares")
    print("SPACE=capture  C=calibrate  U=toggle undistort  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop + flip to match calibration_viewer
        h, w = frame.shape[:2]
        crop_x = (w - h) // 2
        frame = cv2.flip(frame[:, crop_x:crop_x + h], 1)
        img_shape = frame.shape[:2][::-1]  # (w, h)

        display = frame.copy()

        if show_undistort and K is not None:
            display = cv2.undistort(display, K, dist)
            cv2.putText(display, "UNDISTORTED", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect checkerboard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if found:
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, board_size, corners_sub, found)
            cv2.putText(display, "BOARD FOUND — SPACE to capture", (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "No board detected", (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

        # Status
        status = f"Captured: {len(obj_points)}"
        if K is not None:
            status += f"  |  Calibrated (RMS shown in terminal)"
        cv2.putText(display, status, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Camera Intrinsic Calibration", display)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners_sub)
            print(f"Captured frame {len(obj_points)}")
        elif key == ord('c'):
            if len(obj_points) < 5:
                print(f"Need at least 5 frames, have {len(obj_points)}")
            else:
                print(f"Running calibration on {len(obj_points)} frames...")
                K, dist, rms = calibrate(obj_points, img_points, img_shape)
                print(f"  RMS reprojection error: {rms:.4f} px  (aim for < 1.0)")
                print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
                print(f"  dist: {dist.ravel().tolist()}")
                data = {
                    "camera_matrix":    K.tolist(),
                    "dist_coeffs":      dist.tolist(),
                    "rms_error_px":     float(rms),
                    "image_size":       list(img_shape),
                    "board_cols":       args.cols,
                    "board_rows":       args.rows,
                    "square_m":         square_m,
                    "n_frames":         len(obj_points),
                    "saved_at":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                with open(SAVE_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved to {SAVE_PATH}")
        elif key == ord('u'):
            if K is None:
                print("Run calibration first (press C)")
            else:
                show_undistort = not show_undistort
                print(f"Undistort preview: {'ON' if show_undistort else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
