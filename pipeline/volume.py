"""
Volume estimation pipeline (synced with notebook v3.1).
Functions: clean_food_mask, detect_plate, estimate_volumes, classify_portion.
"""
import numpy as np
import cv2
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    PLATE_DIAMETER_CM, EXPECTED_MAX_FOOD_HEIGHT_CM,
    UNIVERSAL_MAX_HEIGHT_CM, PX_PER_CM_MIN, PX_PER_CM_MAX,
    STARCH_REFERENCE_VOLUME_CM3,
)

# v3.1: Density read from NUTRITION_DB at runtime.
# This is the fallback for foods not in the DB.
DEFAULT_DENSITY = 0.60

MAX_ITEM_WEIGHT_G = {
    'Jollof_Rice': 400, 'Waakye': 400, 'Banku': 350, 'Fufu': 350,
    'Plain_Rice':  350, 'Fried_Plantain': 150, 'Grilled_Chicken': 300,
    'Tilapia':     350, 'Fried_Fish': 250, 'Okro_Soup': 300,
    'light_soup':  300, 'Shito': 60,  'Beans': 300,
    'Boiled_Egg':  65,  'Salad': 200, 'default': 350,
}

PORTION_THRESHOLDS = {
    'small':       0.50,
    'appropriate': 1.00,
    'reduce':      1.50,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_food_mask(raw_mask, raw_depth, plate_baseline, erode_px=2,
                    min_pixels=10):
    """
    Tighten a YOLO segmentation mask. v3.1 fallback chain:
      1. eroded + depth-filtered  (best quality)
      2. raw mask + depth-filtered  (if eroded yields < min_pixels)
      3. eroded mask only  (if depth filter kills everything)
      4. raw mask  (last resort for tiny/thin foods)

    Returns (cleaned_mask: bool ndarray, cleanup_method: str).
    """
    mask_bool = (raw_mask > 0).astype(np.uint8)

    eroded = mask_bool.copy()
    if erode_px > 0:
        k      = erode_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        eroded = cv2.erode(mask_bool, kernel, iterations=1)

    tolerance   = max(plate_baseline * 0.02, 1.0)
    above_plate = (raw_depth > plate_baseline + tolerance).astype(np.uint8)

    # Fallback chain
    full_clean = (eroded & above_plate).astype(bool)
    if np.sum(full_clean) >= min_pixels:
        return full_clean, 'full'

    raw_depth_only = (mask_bool & above_plate).astype(bool)
    if np.sum(raw_depth_only) >= min_pixels:
        return raw_depth_only, 'uneroded_depth'

    eroded_bool = eroded.astype(bool)
    if np.sum(eroded_bool) >= min_pixels:
        return eroded_bool, 'eroded_only'

    return mask_bool.astype(bool), 'raw'


def _diameter_from_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return float(np.sqrt(max(np.sum(mask), 1) / np.pi) * 2)
    _, r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return float(2 * r)


def detect_plate(image, raw_dets, raw_depth, plate_class='Plate'):
    """
    3-tier plate detection: YOLO -> Hough -> depth histogram.
    raw_dets: list of dicts with 'name' and 'mask' keys.
    Returns (plate_mask, plate_baseline, px_per_cm, method).
    """
    img_h, img_w = image.shape[:2]
    min_dim = min(img_h, img_w)
    _yolo_baseline = None

    # Tier 1: YOLO plate class
    for d in raw_dets:
        if d['name'] == plate_class:
            pm = d['mask'].astype(bool)
            if pm.sum() > 100:
                baseline = float(np.median(raw_depth[pm]))
                diam = _diameter_from_mask(d['mask'])
                ppc = diam / PLATE_DIAMETER_CM
                if PX_PER_CM_MIN <= ppc <= PX_PER_CM_MAX:
                    return pm, baseline, ppc, 'yolo'
                _yolo_baseline = baseline

    # Tier 2: Hough circles
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min_dim // 2, param1=50, param2=30,
        minRadius=min_dim // 6, maxRadius=min_dim // 2,
    )
    if circles is not None:
        cx, cy, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        ppc = (2 * r) / PLATE_DIAMETER_CM
        if PX_PER_CM_MIN <= ppc <= PX_PER_CM_MAX:
            pm_u8 = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.circle(pm_u8, (int(cx), int(cy)), int(r), 1, -1)
            pm = pm_u8.astype(bool)
            baseline = _yolo_baseline if _yolo_baseline else float(np.median(raw_depth[pm]))
            return pm, baseline, ppc, 'hough'

    # Tier 3: depth histogram mode
    hist, edges = np.histogram(raw_depth.flatten(), bins=256)
    idx = int(np.argmax(hist))
    mode = float((edges[idx] + edges[idx + 1]) / 2) or float(np.median(raw_depth))
    pm = np.abs(raw_depth - mode) < (mode * 0.05)
    baseline = _yolo_baseline if _yolo_baseline else mode
    r_px = float(np.sqrt(max(pm.sum(), 1) / np.pi))
    ppc  = max((2 * r_px) / PLATE_DIAMETER_CM, 1.0)
    return pm, baseline, ppc, 'depth_fallback'


def estimate_volumes(image_bgr, raw_dets, depth_map, plate_class='Plate',
                     nutrition_db=None, debug=False):
    """
    Fuse segmentation masks with depth map to estimate volume and weight.

    raw_dets: list of dicts with keys: name, mask (H x W uint8), conf.
    nutrition_db: v3.1 unified density source (optional, falls back to DEFAULT_DENSITY).
    Returns list of dicts: name, conf, area_cm2, height_cm, volume_cm3, weight_g.
    """
    plate_mask, plate_baseline, px_per_cm, method = detect_plate(
        image_bgr, raw_dets, depth_map, plate_class
    )
    pixel_area_cm2 = (1.0 / px_per_cm) ** 2

    # Pass 1: clean masks + collect deltas
    cleaned    = []
    all_deltas = []
    for d in raw_dets:
        if d['name'].lower() == 'plate' or d['name'] == plate_class:
            continue
        if int(d['mask'].sum()) == 0:
            continue
        cm, cleanup_method = clean_food_mask(d['mask'], depth_map, plate_baseline)
        if int(cm.sum()) < 10:
            continue
        cleaned.append((d, cm, cleanup_method))
        pos = (depth_map[cm] - plate_baseline)
        pos = pos[pos > 0]
        if len(pos):
            all_deltas.append(pos)

    delta_95 = float(np.percentile(np.concatenate(all_deltas), 95)) \
               if all_deltas else 1.0

    if debug:
        print(f"[VOL] plate={method} px/cm={px_per_cm:.2f} delta_95={delta_95:.1f}")

    results = []
    for d, cm, cleanup_method in cleaned:
        dvals = depth_map[cm] - plate_baseline
        norm_h = np.clip(dvals / max(delta_95, 1e-6), 0, 1.5)
        h_cm   = np.clip(norm_h * EXPECTED_MAX_FOOD_HEIGHT_CM, 0, UNIVERSAL_MAX_HEIGHT_CM)

        vol    = float(np.sum(h_cm) * pixel_area_cm2)
        area   = float(cm.sum() * pixel_area_cm2)

        # v3.1: Read density from NUTRITION_DB, fallback to DEFAULT_DENSITY
        density = DEFAULT_DENSITY
        if nutrition_db:
            db_entry = nutrition_db.get(d['name'].lower(), {})
            density  = db_entry.get('density', DEFAULT_DENSITY)

        raw_w   = vol * density
        max_w   = MAX_ITEM_WEIGHT_G.get(d['name'], MAX_ITEM_WEIGHT_G['default'])
        weight  = min(raw_w, max_w)

        if nutrition_db:
            db_entry = nutrition_db.get(d['name'].lower(), {})
            gi_value = db_entry.get('glycemic_index')
            gi_class = db_entry.get('gi_classification')
            per100   = db_entry.get('per_100g', {})
            carbs_g  = (weight / 100.0) * per100.get('carbs', 0)
        else:
            gi_value = None
            gi_class = None
            carbs_g  = 0.0
        gl = (gi_value or 0) * carbs_g / 100

        results.append({
            'name':           d['name'],
            'conf':           d['conf'],
            'mask':           d['mask'],
            'area_cm2':       round(area, 1),
            'height_cm':      round(float(np.mean(h_cm)), 2),
            'volume_cm3':     round(vol, 1),
            'weight_g':       round(weight, 1),
            'cleanup_method': cleanup_method,
            'gi_value':       gi_value,
            'gi_class':       gi_class,
            'carbs_g':        round(carbs_g, 1),
            'glycemic_load':  round(gl, 1),
        })

    return results, {'plate_mask': plate_mask, 'plate_method': method, 'px_per_cm': px_per_cm}


def classify_portion(volume_cm3):
    """
    Compare volume against 225 cm3 medium-orange reference.
    Returns: 'small' | 'appropriate' | 'reduce' | 'excessive'
    """
    ratio = volume_cm3 / STARCH_REFERENCE_VOLUME_CM3
    if   ratio <= PORTION_THRESHOLDS['small']:       return 'small'
    elif ratio <= PORTION_THRESHOLDS['appropriate']: return 'appropriate'
    elif ratio <= PORTION_THRESHOLDS['reduce']:      return 'reduce'
    else:                                            return 'excessive'
