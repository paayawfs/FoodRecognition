"""
Flask Web Interface — Edge AI Food Identification System
Ghanaian Food Recognition with Diabetic Dietary Recommendations (v3.2)
"""

import os
import io
import csv
import json
import base64
import datetime
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory, Response

from config import (
    BASE_DIR, CAPTURES_DIR, EXPORTS_DIR, CONFIDENCE_THRESHOLD,
    MODEL_PATH, NUTRITION_DB_PATH,
)
from database.db import get_db, init_db
from pipeline.camera import Camera
from pipeline.segmentation import FoodSegmenter
from pipeline.depth import DepthEstimator
from pipeline.volume import estimate_volumes
from pipeline.nutrition import (
    load_nutrition_db, calculate_nutrition,
    generate_recommendation, db_key,
)

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# ── lazy model singletons ──────────────────────────────────────────────────────
_segmenter    = None
_depth_est    = None
_nutrition_db = None


def get_segmenter():
    global _segmenter
    if _segmenter is None:
        class_names_path = os.path.join(BASE_DIR, 'database', 'class_names.json')
        # NCNN folder first (Pi deployment), then .pt fallback
        for candidate in [
            MODEL_PATH,                          # models/yolov8s_seg/ — NCNN
            os.path.join('models', 'best.pt'),   # PyTorch fallback
            'yolov8s-seg.pt',                    # pretrained last resort
        ]:
            if os.path.exists(candidate) or candidate == 'yolov8s-seg.pt':
                _segmenter = FoodSegmenter(candidate, class_names_path=class_names_path)
                break
    return _segmenter


def get_depth_estimator():
    global _depth_est
    if _depth_est is None:
        _depth_est = DepthEstimator()
    return _depth_est


def get_nutrition_db():
    global _nutrition_db
    if _nutrition_db is None:
        _nutrition_db = load_nutrition_db()
    return _nutrition_db


# ── startup ───────────────────────────────────────────────────────────────────
Path(CAPTURES_DIR).mkdir(parents=True, exist_ok=True)
Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)
init_db()


# ── core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(image_bgr: np.ndarray) -> dict:
    """
    Full inference pipeline on a BGR numpy image.
    Returns: foods, totals, plate_assessment, annotated_b64, plate_method.
    """
    segmenter = get_segmenter()
    depth_est = get_depth_estimator()
    nutrition = get_nutrition_db()

    # 1. Segmentation — masks only, no bounding boxes in annotated image
    result = segmenter.predict(image_bgr, confidence=CONFIDENCE_THRESHOLD)

    # Build raw_dets list compatible with volume pipeline
    # result.names may be empty for NCNN models — fall back to segmenter.class_names
    names_map = result.names if result.names else {
        i: n for i, n in enumerate(segmenter.class_names)
    }
    raw_dets = []
    if result.masks is not None:
        for box, mask_data in zip(result.boxes, result.masks.data.cpu().numpy()):
            cls_id = int(box.cls[0])
            name   = names_map.get(cls_id, f'class_{cls_id}')
            conf   = float(box.conf[0])
            mask   = cv2.resize(mask_data, (image_bgr.shape[1], image_bgr.shape[0]))
            raw_dets.append({
                'name': name,
                'conf': conf,
                'mask': (mask > 0.5).astype(np.uint8),
            })

    # 2. Depth estimation (RGB input required by MiDaS)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth_map = depth_est.estimate(image_rgb)

    # 3. Volume estimation (v3.1: pass nutrition_db for unified density lookup)
    vol_results, plate_info = estimate_volumes(
        image_bgr, raw_dets, depth_map, nutrition_db=nutrition
    )

    # 4. Build detection dicts for three-level generate_recommendation
    items_for_assess = []
    vol_results_map  = {d['name']: d for d in vol_results}

    for item in vol_results:
        class_name = item['name']
        entry      = nutrition.get(db_key(class_name), {})
        items_for_assess.append({
            'class_name':     class_name,
            'mask':           item['mask'],
            'area_px':        int(item['mask'].sum()),
            'volume_cm3':     item['volume_cm3'],
            'weight_g':       item['weight_g'],
            'gi':             entry.get('glycemic_index', 50),
            'carbs_per_100g': entry.get('per_100g', {}).get('carbs', 0),
        })

    # 5. Three-level plate assessment (v3.3.1)
    plate_area_px = int(plate_info.get('plate_mask', np.zeros(1)).sum())
    rec_result    = generate_recommendation(items_for_assess, plate_area_px)

    # 6. Merge calories/protein/fat (not computed inside generate_recommendation)
    foods_out = []
    for item in rec_result['items']:
        nut  = calculate_nutrition(nutrition, item['class_name'], item.get('weight_g') or 0)
        conf = vol_results_map.get(item['class_name'], {}).get('conf', 0.0)
        foods_out.append({
            **item,
            'confidence': round(conf, 3),
            'calories':   nut['calories'],
            'protein_g':  nut['protein_g'],
            'fat_g':      nut['fat_g'],
        })

    totals = {
        'carbs_g':   rec_result['total_carbs_g'],
        'calories':  round(sum(f.get('calories', 0)  for f in foods_out), 1),
        'protein_g': round(sum(f.get('protein_g', 0) for f in foods_out), 1),
        'fat_g':     round(sum(f.get('fat_g', 0)     for f in foods_out), 1),
    }

    # 7. Annotated image — segmentation masks only, no bounding boxes
    annotated_bgr = result.plot(boxes=False)
    _, buf = cv2.imencode('.jpg', annotated_bgr)
    annotated_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        'foods':            foods_out,
        'totals':           totals,
        'plate_assessment': rec_result['plate_assessment'],
        'recommendations':  rec_result['recommendations'],
        'annotated_b64':    annotated_b64,
        'plate_method':     plate_info.get('plate_method', 'unknown'),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/video_feed')
def video_feed():
    camera = Camera.get_instance()
    return Response(
        camera.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


# ── Users ─────────────────────────────────────────────────────────────────────

@app.route('/api/users', methods=['GET'])
def list_users():
    db   = get_db()
    rows = db.execute("SELECT * FROM users ORDER BY name").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400
    carb_target = float(data.get('daily_carb_target_g', 135.0))
    db = get_db()
    with db:
        cur = db.execute(
            "INSERT INTO users (name, daily_carb_target_g) VALUES (?, ?)",
            (name, carb_target),
        )
    user = dict(db.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone())
    db.close()
    return jsonify(user), 201


@app.route('/api/users/<int:uid>', methods=['GET'])
def get_user(uid):
    db  = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    db.close()
    if not row:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(dict(row))


@app.route('/api/users/<int:uid>', methods=['PUT'])
def update_user(uid):
    data = request.get_json() or {}
    db   = get_db()
    row  = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        db.close()
        return jsonify({'error': 'User not found'}), 404
    name        = (data.get('name') or row['name']).strip()
    carb_target = float(data.get('daily_carb_target_g', row['daily_carb_target_g']))
    with db:
        db.execute(
            "UPDATE users SET name=?, daily_carb_target_g=? WHERE id=?",
            (name, carb_target, uid),
        )
    user = dict(db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
    db.close()
    return jsonify(user)


@app.route('/api/users/<int:uid>', methods=['DELETE'])
def delete_user(uid):
    db = get_db()
    with db:
        db.execute("DELETE FROM users WHERE id=?", (uid,))
    db.close()
    return '', 204


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route('/api/dashboard/<int:uid>')
def dashboard(uid):
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        db.close()
        return jsonify({'error': 'User not found'}), 404
    user  = dict(user)
    today = datetime.date.today().isoformat()

    agg = db.execute(
        """SELECT COALESCE(SUM(total_carbs_g),0) AS today_carbs,
                  COUNT(*) AS today_meals
           FROM meals WHERE user_id=? AND DATE(captured_at)=?""",
        (uid, today),
    ).fetchone()

    recent = db.execute(
        """SELECT m.id, m.captured_at, m.total_carbs_g,
                  m.annotated_image_path,
                  GROUP_CONCAT(mi.food_name, ', ') AS foods
           FROM meals m
           LEFT JOIN meal_items mi ON mi.meal_id = m.id
           WHERE m.user_id=?
           GROUP BY m.id
           ORDER BY m.captured_at DESC LIMIT 5""",
        (uid,),
    ).fetchall()
    db.close()

    today_carbs = round(agg['today_carbs'], 1)
    budget      = user['daily_carb_target_g']

    recent_out = []
    for r in recent:
        rd = dict(r)
        ann = rd.get('annotated_image_path')
        rd['annotated_url'] = (
            '/static/captures/' + os.path.basename(ann)
            if ann and os.path.exists(ann) else None
        )
        del rd['annotated_image_path']
        recent_out.append(rd)

    return jsonify({
        'user':           user,
        'today_carbs':    today_carbs,
        'today_meals':    agg['today_meals'],
        'carb_budget':    budget,
        'carb_remaining': round(max(0.0, budget - today_carbs), 1),
        'recent_meals':   recent_out,
    })


# ── Capture (fast) ─────────────────────────────────────────────────────────────

@app.route('/api/capture', methods=['POST'])
def capture():
    """Grab a still from the camera and save to disk. Returns filename + preview URL."""
    try:
        camera = Camera.get_instance()
        frame  = camera.capture_still()
        if frame is None:
            return jsonify({'error': 'Camera read failed'}), 500

        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
        filename = f"capture_{ts}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)
        cv2.imwrite(filepath, frame)

        return jsonify({
            'filename': filename,
            'url':      f'/static/captures/{filename}',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Upload image ──────────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Accept an uploaded image file, save to captures dir, return filename."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Unsupported image format'}), 400

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
    filename = f"upload_{ts}{ext}"
    filepath = os.path.join(CAPTURES_DIR, filename)
    file.save(filepath)

    return jsonify({
        'filename': filename,
        'url':      f'/static/captures/{filename}',
    })


# ── Infer (slow) ───────────────────────────────────────────────────────────────

@app.route('/api/infer', methods=['POST'])
def infer():
    """Run the full ML pipeline on a previously captured image."""
    data     = request.get_json() or {}
    filename = data.get('filename')
    user_id  = data.get('user_id')
    notes    = data.get('notes', '')

    if not filename:
        return jsonify({'error': 'filename required'}), 400

    filepath = os.path.join(CAPTURES_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image not found — did capture succeed?'}), 404

    try:
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            return jsonify({'error': 'Cannot decode image'}), 500

        result = run_pipeline(image_bgr)

        # Persist annotated image alongside original
        ann_filename = 'ann_' + filename
        ann_path     = os.path.join(CAPTURES_DIR, ann_filename)
        ann_bytes    = base64.b64decode(result['annotated_b64'])
        with open(ann_path, 'wb') as f:
            f.write(ann_bytes)

        # Persist meal record if a user is logged in
        meal_id = None
        if user_id:
            db      = get_db()
            pa_json  = json.dumps(result['plate_assessment'])
            rec_json = json.dumps(result['recommendations'])
            with db:
                cur = db.execute(
                    """INSERT INTO meals
                       (user_id, original_image_path, annotated_image_path,
                        total_carbs_g, plate_assessment, recommendations_json, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, filepath, ann_path, result['totals']['carbs_g'],
                     pa_json, rec_json, notes),
                )
                meal_id = cur.lastrowid
                for food in result['foods']:
                    db.execute(
                        """INSERT INTO meal_items
                           (meal_id, food_name, confidence, portion_category,
                            estimated_volume_cm3, estimated_weight_g, carbs_g,
                            glycemic_index, gi_classification, recommendation)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            meal_id,
                            food['food_name'],
                            food['confidence'],
                            food['portion_category'],
                            food['volume_cm3'],
                            food['weight_g'],
                            food['carbs_g'],
                            food['glycemic_index'],
                            food['gi_classification'],
                            food['recommendation'],
                        ),
                    )
            db.close()

        return jsonify({
            'meal_id':          meal_id,
            'foods':            result['foods'],
            'totals':           result['totals'],
            'plate_assessment': result['plate_assessment'],
            'recommendations':  result['recommendations'],
            'annotated_url':    f'/static/captures/{ann_filename}',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Meal history ───────────────────────────────────────────────────────────────

@app.route('/api/meals/<int:uid>')
def meal_history(uid):
    limit  = min(max(int(request.args.get('limit',  20)), 1), 100)
    offset = max(int(request.args.get('offset',  0)), 0)
    db     = get_db()
    rows   = db.execute(
        """SELECT m.id, m.captured_at, m.total_carbs_g, m.notes,
                  m.annotated_image_path,
                  GROUP_CONCAT(mi.food_name, ', ') AS foods
           FROM meals m
           LEFT JOIN meal_items mi ON mi.meal_id = m.id
           WHERE m.user_id=?
           GROUP BY m.id
           ORDER BY m.captured_at DESC
           LIMIT ? OFFSET ?""",
        (uid, limit, offset),
    ).fetchall()
    db.close()

    out = []
    for r in rows:
        rd  = dict(r)
        ann = rd.pop('annotated_image_path', None)
        rd['annotated_url'] = (
            '/static/captures/' + os.path.basename(ann)
            if ann and os.path.exists(ann) else None
        )
        out.append(rd)
    return jsonify(out)


@app.route('/api/meals/detail/<int:meal_id>')
def meal_detail(meal_id):
    db   = get_db()
    meal = db.execute("SELECT * FROM meals WHERE id=?", (meal_id,)).fetchone()
    if not meal:
        db.close()
        return jsonify({'error': 'Meal not found'}), 404
    meal  = dict(meal)
    items = [dict(r) for r in
             db.execute("SELECT * FROM meal_items WHERE meal_id=? ORDER BY id",
                        (meal_id,)).fetchall()]
    db.close()

    # Deserialize plate_assessment and recommendations from JSON strings
    pa_raw  = meal.pop('plate_assessment', None)
    rec_raw = meal.pop('recommendations_json', None)
    meal['plate_assessment'] = json.loads(pa_raw)  if pa_raw  else {}
    meal['recommendations']  = json.loads(rec_raw) if rec_raw else {}

    orig = meal.pop('original_image_path', None)
    meal['original_url'] = (
        '/static/captures/' + os.path.basename(orig)
        if orig and os.path.exists(orig) else None
    )
    ann = meal.pop('annotated_image_path', None)
    meal['annotated_url'] = (
        '/static/captures/' + os.path.basename(ann)
        if ann and os.path.exists(ann) else None
    )
    return jsonify({'meal': meal, 'items': items})


@app.route('/api/meals/<int:meal_id>', methods=['DELETE'])
def delete_meal(meal_id):
    db = get_db()
    with db:
        db.execute("DELETE FROM meal_items WHERE meal_id=?", (meal_id,))
        db.execute("DELETE FROM meals WHERE id=?", (meal_id,))
    db.close()
    return '', 204


# ── Export ────────────────────────────────────────────────────────────────────

@app.route('/api/export/csv/<int:uid>')
def export_csv(uid):
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        db.close()
        return jsonify({'error': 'User not found'}), 404
    rows = db.execute(
        """SELECT m.captured_at, mi.food_name, mi.portion_category,
                  mi.estimated_weight_g, mi.carbs_g, mi.glycemic_index,
                  mi.gi_classification, mi.recommendation
           FROM meals m
           JOIN meal_items mi ON mi.meal_id = m.id
           WHERE m.user_id=?
           ORDER BY m.captured_at DESC""",
        (uid,),
    ).fetchall()
    db.close()

    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(['Date/Time', 'Food', 'Portion', 'Weight (g)',
                'Carbs (g)', 'GI', 'GI Class', 'Recommendation'])
    for r in rows:
        w.writerow(list(r))
    buf   = io.BytesIO(si.getvalue().encode('utf-8'))
    si.close()
    fname = (f"meals_{dict(user)['name'].replace(' ', '_')}"
             f"_{datetime.date.today()}.csv")
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True, download_name=fname)


@app.route('/api/export/pdf/<int:uid>')
def export_pdf(uid):
    """Export meal report as PDF (fpdf2); falls back to printable HTML."""
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        db.close()
        return jsonify({'error': 'User not found'}), 404
    user = dict(user)
    rows = db.execute(
        """SELECT m.captured_at, m.total_carbs_g,
                  GROUP_CONCAT(mi.food_name, ', ') AS foods
           FROM meals m
           LEFT JOIN meal_items mi ON mi.meal_id = m.id
           WHERE m.user_id=?
           GROUP BY m.id
           ORDER BY m.captured_at DESC""",
        (uid,),
    ).fetchall()
    db.close()

    try:
        from fpdf import FPDF  # fpdf2
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, f"Meal Report — {user['name']}", new_line='NEXT')
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f"Generated: {datetime.date.today()}", new_line='NEXT')
        pdf.ln(4)
        for r in rows:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 7, r['captured_at'], new_line='NEXT')
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(0, 6, f"Foods: {r['foods'] or '—'}")
            pdf.cell(0, 6, f"Total carbs: {r['total_carbs_g'] or 0:.1f} g",
                     new_line='NEXT')
            pdf.ln(3)
        buf   = io.BytesIO(bytes(pdf.output()))
        fname = f"report_{user['name'].replace(' ', '_')}_{datetime.date.today()}.pdf"
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name=fname)

    except ImportError:
        # Graceful degradation: printable HTML
        rows_html = ''.join(
            f"<tr><td>{r['captured_at']}</td><td>{r['foods'] or '—'}</td>"
            f"<td>{r['total_carbs_g'] or 0:.1f}</td></tr>"
            for r in rows
        )
        html = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><title>Meal Report — {user['name']}</title>
<style>
  body{{font-family:sans-serif;padding:2rem;max-width:800px;margin:auto}}
  h1{{color:#2563EB}} table{{border-collapse:collapse;width:100%;margin-top:1rem}}
  th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
  th{{background:#2563EB;color:#fff}}
  @media print{{body{{padding:0}}}}
</style></head>
<body>
<h1>Meal Report — {user['name']}</h1>
<p>Generated: {datetime.date.today()}</p>
<table>
  <thead><tr><th>Date / Time</th><th>Foods</th><th>Carbs (g)</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</body></html>"""
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\nGhanaian Food Analysis System — Flask SPA v3.2")
    print("=" * 48)
    print(f"  Captures : {CAPTURES_DIR}")
    print(f"  Exports  : {EXPORTS_DIR}")
    print("  Open     : http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
