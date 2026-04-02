"""
Nutrition database lookup and recommendation generation (v3.3.1).
Three-level Diabetic Plate Model: area composition / starch portion / glycemic load.
"""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NUTRITION_DB_PATH

# v3.3.1: Moulded starches use orange-reference volume; spread starches use GDA weight.
FOOD_CATEGORIES = {
    'Salad':           'vegetable',
    'Tilapia':         'protein',
    'Grilled_Chicken': 'protein',
    'Fried_Fish':      'protein',
    'Beans':           'protein',
    'Boiled_Egg':      'protein',
    'Fufu':            'starch_moulded',
    'Banku':           'starch_moulded',
    'Rice_Balls':      'starch_moulded',
    'Tuo_Zaafi':       'starch_moulded',
    'Jollof_Rice':     'starch_spread',
    'Waakye':          'starch_spread',
    'Plain_Rice':      'starch_spread',
    'Fried_Plantain':  'starch_spread',
    'Okro_Soup':       'soup_sauce',
    'Light_Soup':      'soup_sauce',
    'Shito':           'soup_sauce',
    'Plate':           'ignored',
}

# Backward-compat alias (used by generate_item_recommendation)
PLATE_CATEGORY = FOOD_CATEGORIES

# GDA standard serving weights for spread starches (grams).
# Source: Ghana Dietetic Association Standard Serving Sizes (Lartey et al. 1999).
GDA_SERVING_G = {
    'Jollof_Rice':    180,
    'Waakye':         170,
    'Plain_Rice':     180,
    'Fried_Plantain': 120,
}

ORANGE_REFERENCE_CM3 = 220   # medium orange, dietician-confirmed
GL_LOW_MAX    = 10            # GL < 10  → Low
GL_MEDIUM_MAX = 20            # GL 10–19 → Medium  (≥ 20 → High)


def load_nutrition_db(path=None):
    p = path or NUTRITION_DB_PATH
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: Could not load nutrition DB from {p}: {e}")
        return {}


def db_key(class_name: str) -> str:
    return class_name.lower()


def lookup_food(nutrition_db: dict, class_name: str) -> dict:
    return nutrition_db.get(db_key(class_name), {})


def calculate_nutrition(nutrition_db: dict, class_name: str, weight_g: float) -> dict:
    entry = lookup_food(nutrition_db, class_name)
    if not entry:
        return {'carbs_g': 0.0, 'calories': 0.0, 'protein_g': 0.0, 'fat_g': 0.0}
    p    = entry.get('per_100g', {})
    mult = weight_g / 100.0
    return {
        'carbs_g':    round(p.get('carbs',    0) * mult, 1),
        'calories':   round(p.get('calories', 0) * mult, 1),
        'protein_g':  round(p.get('protein',  0) * mult, 1),
        'fat_g':      round(p.get('fat',       0) * mult, 1),
    }


def generate_item_recommendation(class_name: str, portion_category: str,
                                  gi_classification: str) -> str:
    cat      = PLATE_CATEGORY.get(class_name, 'unknown')
    food_lbl = class_name.replace('_', ' ').title()
    portion_category = portion_category or 'appropriate'

    if cat in ('starch_moulded', 'starch_spread'):
        text = _STARCH_REC.get(portion_category, '').format(food=food_lbl)
        if gi_classification == 'High':
            text += ' Pair with vegetables to slow glucose release.'
    elif cat == 'protein':
        text = _PROTEIN_REC.get(portion_category, '').format(food=food_lbl)
    elif cat == 'vegetable':
        text = _VEG_REC.get(portion_category, '').format(food=food_lbl)
    elif cat == 'soup_sauce':
        text = f'{food_lbl} detected as accompaniment.'
    else:
        text = f'{food_lbl} detected.'
    return text


# ── LCD display strings for Raspberry Pi ─────────────────────────────────────

LCD_MESSAGES = {
    'good':    ('Plate Balanced',  'Good portion!'),
    'caution': ('Add Vegetables',  'or Reduce starch'),
    'warning': ('Too Much Starch', 'Reduce portion!'),
}


# ── Three-level recommendation engine (v3.3.1) ───────────────────────────────

def generate_recommendation(detections, plate_area_px=0, nutrition_db=None):
    """
    Three-level Diabetic Plate Model (v3.3.1).

    detections: list of dicts, each with:
        class_name    (str)
        area_px       (int)    — mask pixel count
        volume_cm3    (float)
        weight_g      (float)
        gi            (int)    — glycemic index from nutrition DB
        carbs_per_100g (float) — carbs per 100 g from nutrition DB

    Returns dict: plate_assessment, recommendations (level1/level2/level3),
                  items, total_carbs_g.
    """

    # ── Categorise ────────────────────────────────────────────────────────────
    for det in detections:
        det['category'] = FOOD_CATEGORIES.get(det['class_name'], 'unknown')

    countable  = [d for d in detections
                  if d['category'] not in ('soup_sauce', 'ignored', 'unknown')]
    total_area = sum(d['area_px'] for d in countable) or 1

    area_by_cat = {'vegetable': 0.0, 'protein': 0.0,
                   'starch_moulded': 0.0, 'starch_spread': 0.0}
    for d in countable:
        if d['category'] in area_by_cat:
            area_by_cat[d['category']] += d['area_px']

    veg_ratio     = area_by_cat['vegetable'] / total_area
    protein_ratio = area_by_cat['protein']   / total_area
    starch_ratio  = (area_by_cat['starch_moulded'] +
                     area_by_cat['starch_spread'])  / total_area

    ratios = {
        'vegetable': round(veg_ratio, 3),
        'protein':   round(protein_ratio, 3),
        'starch':    round(starch_ratio, 3),
    }

    # ── Level 1: Plate composition message ───────────────────────────────────
    veg_pct = veg_ratio * 100
    if veg_pct < 30:
        level1_msg  = ('Your plate needs more vegetables. '
                       'Aim for at least half your plate to be salad or fibre-rich foods.')
        alert_level = 'warning'
    elif veg_pct < 50:
        level1_msg  = 'Good start — try to push vegetables to at least half your plate.'
        alert_level = 'caution'
    else:
        level1_msg  = 'Great plate balance.'
        alert_level = 'good'

    detail_messages = []
    if starch_ratio > 0.35:
        detail_messages.append(
            'Your starch portion is taking up a large share of the plate.')
    if protein_ratio < 0.15 and countable:
        detail_messages.append(
            'Consider adding a protein source like fish, chicken, or beans.')

    # ── Level 2: Per-starch portion check ────────────────────────────────────
    level2_results = []

    for d in detections:
        cat  = d['category']
        name = d['class_name']
        disp = name.replace('_', ' ')

        if cat == 'starch_moulded':
            vol = d.get('volume_cm3') or 0.0
            if vol <= 110:
                portion_cat = 'small'
                msg = (f'Your {disp} portion is well within a healthy range. '
                       'Consider adding more vegetables.')
            elif vol <= ORANGE_REFERENCE_CM3:
                portion_cat = 'appropriate'
                msg = (f'Your {disp} portion looks appropriate — '
                       'about the size of a medium orange.')
            else:
                portion_cat = 'reduce'
                msg = (f'Your {disp} portion appears larger than a medium orange '
                       f'(~{ORANGE_REFERENCE_CM3} ml). Consider reducing it slightly.')
            level2_results.append({
                'food':        disp,
                'starch_type': 'moulded',
                'volume_cm3':  round(vol, 1),
                'gda_g':       None,
                'weight_g':    d.get('weight_g'),
                'portion_cat': portion_cat,
                'message':     msg,
            })

        elif cat == 'starch_spread':
            gda = GDA_SERVING_G.get(name)
            wt  = d.get('weight_g') or 0.0
            if gda is None:
                continue
            if wt <= gda:
                portion_cat = 'appropriate'
                msg = f'Your {disp} portion is within a standard serving.'
            else:
                portion_cat = 'reduce'
                msg = (f'Your {disp} portion exceeds a standard serving '
                       f'(~{gda}g). Consider a smaller amount.')
            level2_results.append({
                'food':        disp,
                'starch_type': 'spread',
                'volume_cm3':  None,
                'gda_g':       gda,
                'weight_g':    round(wt, 1),
                'portion_cat': portion_cat,
                'message':     msg,
            })

    # ── Level 3: Glycemic Load per item ──────────────────────────────────────
    level3_results = []

    for d in detections:
        if d['category'] in ('soup_sauce', 'ignored', 'unknown'):
            continue
        gi             = d.get('gi')
        carbs_per_100g = d.get('carbs_per_100g')
        weight_g       = d.get('weight_g') or 0.0
        disp           = d['class_name'].replace('_', ' ')

        if gi is None or carbs_per_100g is None or weight_g == 0:
            gl, gl_level = None, 'unknown'
            gl_msg = 'Glycemic load could not be calculated for this item.'
        else:
            gl = round((gi / 100.0) * carbs_per_100g * (weight_g / 100.0), 1)
            if gl < GL_LOW_MAX:
                gl_level = 'low'
                gl_msg   = 'Low glycemic load — a steady energy release.'
            elif gl < GL_MEDIUM_MAX:
                gl_level = 'medium'
                gl_msg   = 'Moderate glycemic load — manageable in a balanced meal.'
            else:
                gl_level = 'high'
                gl_msg   = 'High glycemic load — this item may cause a blood sugar spike.'

        level3_results.append({
            'food':     disp,
            'gl':       gl,
            'gl_level': gl_level,
            'message':  gl_msg,
        })

    # ── Per-item output (food cards) ──────────────────────────────────────────
    l2_by_name = {r['food']: r for r in level2_results}
    items_out  = []

    for d in detections:
        if d['category'] == 'ignored':
            continue
        disp  = d['class_name'].replace('_', ' ')
        gi    = d.get('gi')
        gi_cls = (None        if gi is None
                  else 'Low'  if gi < 55
                  else 'Medium' if gi < 70
                  else 'High')
        l2   = l2_by_name.get(disp, {})
        wt   = d.get('weight_g') or 0.0
        c100 = d.get('carbs_per_100g') or 0.0
        items_out.append({
            'class_name':        d['class_name'],
            'food_name':         d['class_name'],  # underscore form — consistent with DB
            'category':          d['category'],
            'area_pct':          round((d['area_px'] / total_area) * 100, 1)
                                 if d['category'] != 'soup_sauce' else None,
            'volume_cm3':        round(d['volume_cm3'], 1)
                                 if d.get('volume_cm3') is not None else None,
            'weight_g':          round(wt, 1) if wt else None,
            'carbs_g':           round(c100 * wt / 100, 1),
            'glycemic_index':    gi,
            'gi_classification': gi_cls,
            'portion_category':  l2.get('portion_cat'),
            'recommendation':    l2.get('message'),
        })

    total_carbs_g = round(sum(i['carbs_g'] for i in items_out), 1)
    lcd           = LCD_MESSAGES.get(alert_level, LCD_MESSAGES['caution'])

    return {
        'plate_assessment': {
            'alert_level':     alert_level,
            'overall_message': level1_msg,
            'detail_messages': detail_messages,
            'ratios':          ratios,
            'lcd':             lcd,
        },
        'recommendations': {
            'level1_message': level1_msg,
            'level2':         level2_results,
            'level3':         level3_results,
        },
        'items':          items_out,
        'total_carbs_g':  total_carbs_g,
    }


# Backward-compatible alias
assess_plate = generate_recommendation
