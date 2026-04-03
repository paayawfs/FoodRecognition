"""
FoodAI depth calibration (v1.0).

Place a boiled egg upright in the centre of the plate.
Hold the Pi camera ~45 cm directly above, steady and level.
Run:  python calibrate.py

The script captures a still, runs MiDaS, detects the plate baseline,
finds the egg peak depth delta, and computes DEPTH_UNITS_PER_CM —
the number of MiDaS depth units corresponding to 1 cm of real height.

Output: the two lines to paste into config.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from pipeline.camera import Camera
from pipeline.depth  import DepthEstimator
from pipeline.volume import detect_plate
from config import CAPTURES_DIR

EGG_HEIGHT_CM = 5.5   # large boiled egg standing upright (short axis up)


def main():
    print("FoodAI Calibration Script")
    print("=" * 44)
    print(f"Reference object: boiled egg, {EGG_HEIGHT_CM} cm tall")
    print()

    # 1. Capture still
    print("Capturing still image...")
    cam       = Camera.get_instance()
    frame_bgr = cam.capture_still()
    if frame_bgr is None:
        print("ERROR: capture failed.")
        sys.exit(1)
    out = os.path.join(CAPTURES_DIR, 'calibration_capture.jpg')
    cv2.imwrite(out, frame_bgr)
    print(f"Image saved: {out}")
    print("Check this image looks correct before trusting the result.\n")

    # 2. Depth estimation
    print("Running MiDaS depth estimation...")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    estimator = DepthEstimator()
    depth_map = estimator.estimate(frame_rgb)

    # 3. Plate detection — gives baseline and px_per_cm
    # Pass empty dets list: forces Hough or depth-histogram fallback
    print("Detecting plate...")
    plate_mask, plate_baseline, px_per_cm, method = detect_plate(
        frame_bgr, [], depth_map
    )
    print(f"  Plate method : {method}")
    print(f"  px_per_cm    : {px_per_cm:.3f}")
    print(f"  Baseline     : {plate_baseline:.4f}\n")

    # 4. Find egg peak
    # The egg is the tallest object above the plate.
    # Use the 99th percentile of all above-plate depth deltas as the egg peak.
    above = depth_map - plate_baseline
    pos   = above[above > 0].flatten()

    if len(pos) < 50:
        print("ERROR: Too few pixels above plate baseline.")
        print("Make sure the plate and egg are clearly visible.")
        sys.exit(1)

    delta_99 = float(np.percentile(pos, 99))
    delta_95 = float(np.percentile(pos, 95))
    print(f"Depth delta 95th pct : {delta_95:.4f}")
    print(f"Depth delta 99th pct : {delta_99:.4f}  (egg peak estimate)")

    # 5. Compute DEPTH_UNITS_PER_CM
    # delta_99 MiDaS units correspond to EGG_HEIGHT_CM real cm
    depth_units_per_cm = delta_99 / EGG_HEIGHT_CM
    estimated_egg_h    = delta_99 / depth_units_per_cm  # sanity: should equal EGG_HEIGHT_CM

    print(f"\nComputed DEPTH_UNITS_PER_CM : {depth_units_per_cm:.4f}")
    print(f"Estimated egg height        : {estimated_egg_h:.2f} cm  (expected {EGG_HEIGHT_CM})")

    # 6. Print result
    print()
    print("=" * 44)
    print("RESULT — paste these two lines into config.py")
    print("(replacing the current USE_CALIBRATED_DEPTH and DEPTH_UNITS_PER_CM lines):")
    print("=" * 44)
    print(f"USE_CALIBRATED_DEPTH = True")
    print(f"DEPTH_UNITS_PER_CM   = {round(depth_units_per_cm, 4)}")
    print()
    print("Then restart: sudo pkill -f app.py && python app.py")


if __name__ == '__main__':
    main()
