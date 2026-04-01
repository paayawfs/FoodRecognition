"""
Nutrition database lookup and recommendation generation (v3.3).
"""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NUTRITION_DB_PATH

# v3.3: starch_spread now assessed via GDA serving weights (not just plate ratio).
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
    'Light_Soup':      'soup_sauce',
    'Shito':           'soup_sauce',
    'Salad':           'vegetable',
}

# GDA serving thresholds (number of servings, where 1 serving = 20g carbs).
# Source: Ghana Dietetic Association Standard Serving Sizes (Lartey et al. 1999).
GDA_SPREAD_THRESHOLDS = {
    'small':       0.75,
    'appropriate': 2.00,   # up to 2 servings = ~40g carbs from starch
    'reduce':      3.00,
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


def classify_starch_portion(items, nutrition_db=None):
    """
    Classify starch portions using two reference systems (v3.3).

    - starch_moulded (banku, fufu): volume vs 220 cm³ medium orange.
    - starch_spread (jollof, rice, plantain): weight vs GDA serving sizes.

    items: list of dicts with 'class_name', 'volume_cm3', 'weight_g'.
    nutrition_db: loaded nutrition DB dict (loaded from file if not provided).
    """
    from config import STARCH_REFERENCE_VOLUME_CM3
    from pipeline.volume import PORTION_THRESHOLDS

    if nutrition_db is None:
        nutrition_db = load_nutrition_db()

    moulded_volume        = 0.0
    moulded_foods         = []
    spread_weight         = 0.0
    spread_foods          = []
    spread_servings_total = 0.0

    for item in items:
        cat = PLATE_CATEGORY.get(item['class_name'], 'unknown')
        if cat == 'starch_moulded':
            moulded_volume += item.get('volume_cm3', 0.0)
            moulded_foods.append(item['class_name'])
        elif cat == 'starch_spread':
            w = item.get('weight_g', 0.0)
            spread_weight += w
            spread_foods.append(item['class_name'])
            db_entry = nutrition_db.get(db_key(item['class_name']), {})
            gda_g = db_entry.get('gda_serving_g')
            if gda_g and gda_g > 0:
                spread_servings_total += w / gda_g

    # ── Path 1: Moulded starches (volume-based, orange reference) ─────────────
    if moulded_volume > 0:
        ratio = moulded_volume / STARCH_REFERENCE_VOLUME_CM3
        if   ratio <= PORTION_THRESHOLDS['small']:       category = 'small'
        elif ratio <= PORTION_THRESHOLDS['appropriate']:  category = 'appropriate'
        elif ratio <= PORTION_THRESHOLDS['reduce']:       category = 'reduce'
        else:                                             category = 'excessive'

        food_label = ', '.join(f.replace('_', ' ').title() for f in moulded_foods)
        _MOULDED_MESSAGES = {
            'small':       f'Your {food_label} portion is small.',
            'appropriate': f'Your {food_label} portion looks good.',
            'reduce':      (f'Consider reducing your {food_label}. '
                            'Aim for about the size of a medium orange.'),
            'excessive':   (f'Your {food_label} portion is too large. '
                            'It should be about the size of a medium orange.'),
        }
        return {
            'total_starch_volume_cm3': moulded_volume,
            'total_starch_weight_g':   0.0,
            'portion_category':        category,
            'ratio_to_reference':      round(ratio, 2),
            'reference_type':          'volume',
            'message':                 _MOULDED_MESSAGES[category],
        }

    # ── Path 2: Spread starches (weight-based, GDA serving reference) ─────────
    if spread_weight > 0:
        if   spread_servings_total <= GDA_SPREAD_THRESHOLDS['small']:       category = 'small'
        elif spread_servings_total <= GDA_SPREAD_THRESHOLDS['appropriate']:  category = 'appropriate'
        elif spread_servings_total <= GDA_SPREAD_THRESHOLDS['reduce']:       category = 'reduce'
        else:                                                                category = 'excessive'

        food_label   = ', '.join(f.replace('_', ' ').title() for f in spread_foods)
        servings_str = f'{spread_servings_total:.1f}'
        _SPREAD_MESSAGES = {
            'small':       f'Your {food_label} portion is small ({servings_str} servings).',
            'appropriate': f'Your {food_label} portion is appropriate ({servings_str} servings).',
            'reduce':      (f'Consider reducing your {food_label} ({servings_str} servings). '
                            'Aim for about 2 servings per meal.'),
            'excessive':   (f'Your {food_label} portion is too large ({servings_str} servings). '
                            'A single meal should have about 2 servings.'),
        }
        return {
            'total_starch_volume_cm3': 0.0,
            'total_starch_weight_g':   round(spread_weight, 1),
            'portion_category':        category,
            'ratio_to_reference':      round(spread_servings_total, 2),
            'reference_type':          'weight',
            'message':                 _SPREAD_MESSAGES[category],
        }

    # ── Path 3: No starch detected ────────────────────────────────────────────
    return {
        'total_starch_volume_cm3': 0.0,
        'total_starch_weight_g':   0.0,
        'portion_category':        'none',
        'ratio_to_reference':      0.0,
        'reference_type':          'none',
        'message':                 'No starchy food detected.',
    }


def generate_recommendation(items, plate_area_px=0, nutrition_db=None):
    """
    Top-level v3.3 recommendation engine (synced from notebook).

    Combines plate-composition check (area-based) with starch-portion
    check (volume or GDA-weight based) into a single alert + message set.

    v3.3: spread starches assessed via GDA serving weights, not plate ratio only.

    items: list of dicts with keys:
        class_name, mask, volume_cm3, weight_g,
        gi_value (int|None), gi_class (str|None), carbs_g (float).
    plate_area_px: total plate pixel area (from plate_info).

    Returns a dict with:
        ratios, plate_balanced, vegetables_low, alert_level, overall_message,
        detail_messages, starch_assessment, gi_info, lcd.
    """
    plate  = classify_plate_composition(items, plate_area_px)
    starch = classify_starch_portion(items, nutrition_db=nutrition_db)

    gi_info = [
        {'food': d['class_name'], 'gi': d.get('gi_value'), 'gi_class': d.get('gi_class')}
        for d in items if d.get('gi_value') is not None
    ]

    detail_messages = []

    # ── Two-level decision (from notebook v3.3) ───────────────────────────
    if starch['portion_category'] == 'excessive':
        alert_level     = 'warning'
        overall_message = starch['message']
        detail_messages.extend(plate['messages'])

    elif starch['portion_category'] == 'reduce':
        alert_level     = 'caution'
        overall_message = starch['message']
        detail_messages.extend(plate['messages'])

    else:
        # Starch is fine (or none); check plate composition
        if plate['vegetables_low']:
            alert_level     = 'caution'
            overall_message = ('Add more vegetables to your plate '
                               'for better blood sugar control.')
            if starch['portion_category'] == 'small':
                detail_messages.append(
                    "Your starch portion is small — that's fine, "
                    'just fill the gap with vegetables.'
                )
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
