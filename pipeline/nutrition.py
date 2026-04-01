"""
Nutrition database lookup and recommendation generation (v3.1).
"""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NUTRITION_DB_PATH

# v3.1: starch split into moulded (volume-checked) and spread (ratio only).
#        soup_sauce excluded from plate ratio.
PLATE_CATEGORY = {
    'Jollof_Rice':     'starch_spread',
    'Waakye':          'starch_spread',
    'Plain_Rice':      'starch_spread',
    'Fried_Plantain':  'starch_spread',
    'Banku':           'starch_moulded',
    'Fufu':            'starch_moulded',
    'Beans':           'protein',
    'Grilled_Chicken': 'protein',
    'Tilapia':         'protein',
    'Fried_Fish':      'protein',
    'Boiled_Egg':      'protein',
    'Okro_Soup':       'soup_sauce',
    'light_soup':      'soup_sauce',
    'Shito':           'soup_sauce',
    'Salad':           'vegetable',
}

_STARCH_REC = {
    'small':       'Your {food} portion is small.',
    'appropriate': 'Your {food} portion looks good.',
    'reduce':      'Consider reducing your {food}. Aim for about the size of a medium orange.',
    'excessive':   'Your {food} portion is too large. It should be about the size of a medium orange.',
}

_PROTEIN_REC = {
    'small':       'Your {food} is a small serving — a good choice.',
    'appropriate': 'Good {food} portion.',
    'reduce':      'Your {food} portion is generous.',
    'excessive':   'Consider a smaller serving of {food}.',
}

_VEG_REC = {
    'small':       'Add more {food} to help slow glucose absorption.',
    'appropriate': 'Good amount of {food}.',
    'reduce':      'Plenty of {food} — great for blood sugar control.',
    'excessive':   'Plenty of {food} — great for blood sugar control.',
}


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


# ── Plate-level assessment (synced with notebook v3.1) ───────────────────────

def classify_plate_composition(items, plate_area_px=0):
    """
    Assess plate composition against the 50/25/25 Diabetic Plate Model.
    Uses mask pixel AREAS (not volume) for ratio computation.

    v3.1: starch_moulded and starch_spread both count as 'starch'.
          soup_sauce items are excluded from the ratio (accompaniments).

    items: list of dicts with 'class_name' and 'mask' (H x W uint8).
    """
    import numpy as np
    cat_px = {'starch': 0, 'protein': 0, 'vegetable': 0}

    for item in items:
        cat  = PLATE_CATEGORY.get(item['class_name'], 'unknown')
        mask = item.get('mask')
        area = int(mask.sum()) if mask is not None else 0
        if cat in ('starch_moulded', 'starch_spread'):
            cat_px['starch'] += area
        elif cat == 'soup_sauce':
            pass  # excluded from plate ratio
        elif cat == 'mixed':
            cat_px['starch']  += area // 2
            cat_px['protein'] += area // 2
        elif cat in cat_px:
            cat_px[cat] += area

    total_food_px = sum(cat_px.values())
    if total_food_px == 0:
        return {
            'ratios':         {'starch': 0.0, 'protein': 0.0, 'vegetable': 0.0},
            'plate_balanced': False,
            'vegetables_low': True,
            'messages':       ['No food detected on plate.'],
        }

    ratios = {k: v / total_food_px for k, v in cat_px.items()}
    vegetables_low = ratios['vegetable'] < 0.30
    plate_balanced = ratios['vegetable'] >= 0.40 and ratios['starch'] <= 0.35

    messages = []
    if vegetables_low:
        messages.append(
            'Add more vegetables to your plate for better blood sugar control.'
        )
    if ratios['starch'] > 0.40:
        messages.append(
            'Your plate has too much starchy food. '
            'Try to keep starches to about a quarter of your plate.'
        )
    return {
        'ratios': ratios, 'plate_balanced': plate_balanced,
        'vegetables_low': vegetables_low, 'messages': messages,
    }


def classify_starch_portion(items):
    """
    Classify starch VOLUME against the medium-orange reference (220 cm³).

    v3.1: Only starch_moulded (fufu, banku) is volume-compared.
          starch_spread (rice, plantain) returns 'spread_only'.

    items: list of dicts with 'class_name' and 'volume_cm3'.
    """
    from config import STARCH_REFERENCE_VOLUME_CM3
    from pipeline.volume import PORTION_THRESHOLDS

    moulded_volume = 0.0
    moulded_foods  = []
    has_spread     = False

    for item in items:
        cat = PLATE_CATEGORY.get(item['class_name'], 'unknown')
        if cat == 'starch_moulded':
            moulded_volume += item.get('volume_cm3', 0.0)
            moulded_foods.append(item['class_name'])
        elif cat == 'starch_spread':
            has_spread = True
        elif cat == 'mixed':
            moulded_volume += item.get('volume_cm3', 0.0) * 0.5

    if moulded_volume == 0:
        if has_spread:
            return {
                'total_starch_volume_cm3': 0.0,
                'portion_category':        'spread_only',
                'ratio_to_reference':      0.0,
                'message':                 'Starchy foods assessed by plate proportion only.',
            }
        return {
            'total_starch_volume_cm3': 0.0,
            'portion_category':        'none',
            'ratio_to_reference':      0.0,
            'message':                 'No starchy food detected.',
        }

    ratio = moulded_volume / STARCH_REFERENCE_VOLUME_CM3

    if   ratio <= PORTION_THRESHOLDS['small']:       category = 'small'
    elif ratio <= PORTION_THRESHOLDS['appropriate']:  category = 'appropriate'
    elif ratio <= PORTION_THRESHOLDS['reduce']:       category = 'reduce'
    else:                                             category = 'excessive'

    food_label = ', '.join(f.replace('_', ' ').title() for f in moulded_foods)
    _STARCH_MESSAGES = {
        'small':       f'Your {food_label} portion is small.',
        'appropriate': f'Your {food_label} portion looks good.',
        'reduce':      (f'Consider reducing your {food_label}. '
                        'Aim for about the size of a medium orange.'),
        'excessive':   (f'Your {food_label} portion is too large. '
                        'It should be about the size of a medium orange.'),
    }
    return {
        'total_starch_volume_cm3': moulded_volume,
        'portion_category':        category,
        'ratio_to_reference':      round(ratio, 2),
        'message':                 _STARCH_MESSAGES[category],
    }


def generate_recommendation(items, plate_area_px=0):
    """
    Top-level v3.1 recommendation engine (synced from notebook).

    Combines plate-composition check (area-based) with starch-portion
    check (volume-based) into a single alert + message set.

    v3.1: Handles starch_moulded vs starch_spread, soup_sauce exclusion,
          and 'spread_only' starch assessment.

    items: list of dicts with keys:
        class_name, mask, volume_cm3, weight_g,
        gi_value (int|None), gi_class (str|None), carbs_g (float).
    plate_area_px: total plate pixel area (from plate_info).

    Returns a dict with:
        ratios, plate_balanced, vegetables_low, alert_level, overall_message,
        detail_messages, starch_assessment, gi_info, lcd.
    """
    plate  = classify_plate_composition(items, plate_area_px)
    starch = classify_starch_portion(items)

    gi_info = [
        {'food': d['class_name'], 'gi': d.get('gi_value'), 'gi_class': d.get('gi_class')}
        for d in items if d.get('gi_value') is not None
    ]

    detail_messages = []

    # ── Two-level decision (from notebook v3.1) ───────────────────────────
    if starch['portion_category'] == 'excessive':
        alert_level     = 'warning'
        overall_message = starch['message']
        detail_messages.extend(plate['messages'])

    elif starch['portion_category'] == 'reduce':
        alert_level     = 'caution'
        overall_message = starch['message']
        detail_messages.extend(plate['messages'])

    else:
        # Starch is fine (or spread_only/none); check plate composition
        if plate['vegetables_low']:
            alert_level     = 'caution'
            overall_message = ('Add more vegetables to your plate '
                               'for better blood sugar control.')
            if starch['portion_category'] == 'small':
                detail_messages.append(
                    "Your starch portion is small — that's fine, "
                    'just fill the gap with vegetables.'
                )
            elif starch['portion_category'] == 'spread_only':
                detail_messages.append(starch['message'])
            else:
                detail_messages.append(starch['message'])
        else:
            alert_level     = 'good'
            overall_message = 'Your plate looks well balanced.'
            detail_messages.append(starch['message'])

    # ── High-GI context ───────────────────────────────────────────────────
    high_gi = [g for g in gi_info if g['gi_class'] == 'High']
    if high_gi:
        names = ', '.join(g['food'].replace('_', ' ').title() for g in high_gi)
        detail_messages.append(
            f'{names} has a high glycemic index. '
            'Pair with fibre or vegetables to slow glucose absorption.'
        )

    lcd = LCD_MESSAGES.get(alert_level, LCD_MESSAGES['caution'])

    return {
        'ratios':              plate['ratios'],
        'plate_balanced':      plate['plate_balanced'],
        'vegetables_low':      plate['vegetables_low'],
        'alert_level':         alert_level,
        'overall_message':     overall_message,
        'detail_messages':     detail_messages,
        'starch_assessment':   starch,
        'gi_info':             gi_info,
        'lcd':                 lcd,
    }


# Backward-compatible alias
assess_plate = generate_recommendation
