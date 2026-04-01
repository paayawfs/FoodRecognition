"""Tests for the nutrition pipeline module (synced with notebook v3.1)."""
from pipeline.nutrition import (
    db_key, calculate_nutrition, generate_item_recommendation,
    classify_plate_composition, classify_starch_portion,
    generate_recommendation, assess_plate, LCD_MESSAGES,
)
import numpy as np


class TestDbKey:
    def test_lowercase(self):
        assert db_key("Jollof_Rice") == "jollof_rice"

    def test_already_lower(self):
        assert db_key("beans") == "beans"


class TestCalculateNutrition:
    MOCK_DB = {
        "jollof_rice": {
            "per_100g": {"carbs": 40, "calories": 200, "protein": 5, "fat": 6},
        }
    }

    def test_known_food(self):
        result = calculate_nutrition(self.MOCK_DB, "Jollof_Rice", 200)
        assert result["carbs_g"] == 80.0
        assert result["calories"] == 400.0
        assert result["protein_g"] == 10.0
        assert result["fat_g"] == 12.0

    def test_unknown_food_returns_zeros(self):
        result = calculate_nutrition(self.MOCK_DB, "Unknown_Food", 100)
        assert result == {"carbs_g": 0.0, "calories": 0.0, "protein_g": 0.0, "fat_g": 0.0}

    def test_zero_weight(self):
        result = calculate_nutrition(self.MOCK_DB, "Jollof_Rice", 0)
        assert result["carbs_g"] == 0.0


class TestGenerateItemRecommendation:
    def test_starch_appropriate(self):
        rec = generate_item_recommendation("Jollof_Rice", "appropriate", "High")
        assert "looks good" in rec.lower()

    def test_starch_excessive_high_gi(self):
        rec = generate_item_recommendation("Jollof_Rice", "excessive", "High")
        assert "too large" in rec.lower()
        assert "vegetables" in rec.lower()

    def test_moulded_starch(self):
        rec = generate_item_recommendation("Banku", "appropriate", "Medium")
        assert "looks good" in rec.lower()

    def test_protein(self):
        rec = generate_item_recommendation("Grilled_Chicken", "small", "Low")
        assert "chicken" in rec.lower()

    def test_vegetable(self):
        rec = generate_item_recommendation("Salad", "small", "Low")
        assert "add more" in rec.lower()

    def test_soup_sauce(self):
        rec = generate_item_recommendation("Okro_Soup", "appropriate", "Low")
        assert "accompaniment" in rec.lower()

    def test_unknown_category(self):
        rec = generate_item_recommendation("Mystery_Food", "appropriate", "Medium")
        assert "detected" in rec.lower()


# ── Plate composition (area-based) ──────────────────────────────────────────

class TestClassifyPlateComposition:
    def _item(self, name, area_px):
        return {"class_name": name, "mask": np.ones((area_px, 1), dtype=np.uint8)}

    def test_empty(self):
        result = classify_plate_composition([])
        assert result["vegetables_low"] is True
        assert "No food detected" in result["messages"][0]

    def test_balanced(self):
        items = [self._item("Salad", 500), self._item("Grilled_Chicken", 250),
                 self._item("Jollof_Rice", 250)]
        result = classify_plate_composition(items)
        assert result["ratios"]["vegetable"] == 0.5
        assert result["plate_balanced"] is True
        assert result["vegetables_low"] is False
        assert result["messages"] == []

    def test_starch_heavy_messages(self):
        items = [self._item("Jollof_Rice", 600), self._item("Salad", 200),
                 self._item("Tilapia", 200)]
        result = classify_plate_composition(items)
        assert result["ratios"]["starch"] == 0.6
        assert any("too much starchy" in m for m in result["messages"])

    def test_vegetables_low(self):
        items = [self._item("Jollof_Rice", 400), self._item("Grilled_Chicken", 400),
                 self._item("Salad", 100)]
        result = classify_plate_composition(items)
        assert result["vegetables_low"] is True

    def test_soup_sauce_excluded(self):
        """v3.1: soup_sauce items don't count toward plate ratio."""
        items = [self._item("Salad", 500), self._item("Grilled_Chicken", 250),
                 self._item("Banku", 250), self._item("Okro_Soup", 300)]
        result = classify_plate_composition(items)
        # Okro_Soup (300px) excluded — only 1000px counted
        assert result["ratios"]["vegetable"] == 0.5
        assert result["ratios"]["starch"] == 0.25
        assert result["plate_balanced"] is True

    def test_moulded_and_spread_both_count_as_starch(self):
        """v3.1: Both starch subtypes merge into 'starch' for ratio."""
        items = [self._item("Banku", 200), self._item("Jollof_Rice", 200),
                 self._item("Salad", 300), self._item("Grilled_Chicken", 300)]
        result = classify_plate_composition(items)
        assert result["ratios"]["starch"] == 0.4


# ── Starch portion (volume-based) ───────────────────────────────────────────

class TestClassifyStarchPortion:
    def _item(self, name, vol):
        return {"class_name": name, "volume_cm3": vol,
                "mask": np.ones((10, 1), dtype=np.uint8)}

    def test_no_starch(self):
        items = [self._item("Grilled_Chicken", 100)]
        result = classify_starch_portion(items)
        assert result["portion_category"] == "none"

    def test_spread_only(self):
        """v3.1: spread starches return 'spread_only', not volume-checked."""
        items = [self._item("Jollof_Rice", 400)]
        result = classify_starch_portion(items)
        assert result["portion_category"] == "spread_only"
        assert "plate proportion" in result["message"].lower()

    def test_small_moulded(self):
        items = [self._item("Banku", 80)]
        result = classify_starch_portion(items)
        assert result["portion_category"] == "small"

    def test_appropriate_moulded(self):
        items = [self._item("Fufu", 200)]
        result = classify_starch_portion(items)
        assert result["portion_category"] == "appropriate"

    def test_excessive_moulded(self):
        items = [self._item("Banku", 400)]
        result = classify_starch_portion(items)
        assert result["portion_category"] == "excessive"
        assert "too large" in result["message"].lower()

    def test_ratio_to_reference(self):
        items = [self._item("Fufu", 220)]  # exactly 1x reference (220 cm³)
        result = classify_starch_portion(items)
        assert result["ratio_to_reference"] == 1.0

    def test_moulded_food_label(self):
        """v3.1: message uses the actual food name."""
        items = [self._item("Banku", 200)]
        result = classify_starch_portion(items)
        assert "Banku" in result["message"]


# ── Top-level recommendation engine ─────────────────────────────────────────

class TestGenerateRecommendation:
    def _item(self, name, area_px, vol, gi_val=None, gi_cls=None):
        return {
            "class_name": name,
            "mask": np.ones((area_px, 1), dtype=np.uint8),
            "volume_cm3": vol, "weight_g": vol * 0.5,
            "gi_value": gi_val, "gi_class": gi_cls, "carbs_g": 10.0,
        }

    def test_balanced_plate_good(self):
        items = [
            self._item("Salad", 500, 50, 15, "Low"),
            self._item("Grilled_Chicken", 250, 30, 0, "Low"),
            self._item("Jollof_Rice", 250, 100, 70, "High"),
        ]
        result = generate_recommendation(items)
        assert result["alert_level"] == "good"
        assert "balanced" in result["overall_message"].lower()
        assert result["ratios"]["vegetable"] == 0.5

    def test_excessive_moulded_starch_warning(self):
        """v3.1: Only moulded starches trigger volume-based warning."""
        items = [
            self._item("Banku", 500, 400, 50, "Medium"),
            self._item("Salad", 300, 20, 15, "Low"),
            self._item("Tilapia", 200, 30, 0, "Low"),
        ]
        result = generate_recommendation(items)
        assert result["alert_level"] == "warning"
        assert "too large" in result["overall_message"].lower()

    def test_spread_starch_no_volume_warning(self):
        """v3.1: Spread starches assessed by plate ratio, not volume."""
        items = [
            self._item("Jollof_Rice", 250, 400, 70, "High"),
            self._item("Salad", 500, 50, 15, "Low"),
            self._item("Tilapia", 250, 30, 0, "Low"),
        ]
        result = generate_recommendation(items)
        # Even with 400 cm³ volume, Jollof_Rice is spread → no volume warning
        assert result["alert_level"] == "good"
        assert result["starch_assessment"]["portion_category"] == "spread_only"

    def test_low_veg_caution(self):
        items = [
            self._item("Jollof_Rice", 400, 150, 70, "High"),
            self._item("Grilled_Chicken", 400, 50, 0, "Low"),
            self._item("Salad", 100, 10, 15, "Low"),
        ]
        result = generate_recommendation(items)
        assert result["alert_level"] == "caution"
        assert "vegetables" in result["overall_message"].lower()

    def test_high_gi_detail_message(self):
        items = [
            self._item("Salad", 500, 50, 15, "Low"),
            self._item("Grilled_Chicken", 250, 30, 0, "Low"),
            self._item("Jollof_Rice", 250, 100, 70, "High"),
        ]
        result = generate_recommendation(items)
        high_gi_msgs = [m for m in result["detail_messages"] if "glycemic" in m.lower()]
        assert len(high_gi_msgs) == 1
        assert "Jollof Rice" in high_gi_msgs[0]

    def test_gi_info_populated(self):
        items = [
            self._item("Jollof_Rice", 250, 100, 70, "High"),
            self._item("Salad", 500, 50, 15, "Low"),
        ]
        result = generate_recommendation(items)
        assert len(result["gi_info"]) == 2
        assert result["gi_info"][0]["gi"] == 70

    def test_lcd_messages(self):
        """v3.1: Use moulded starch for volume-triggered warning."""
        items = [self._item("Banku", 500, 400, 50, "Medium")]
        result = generate_recommendation(items)
        assert result["lcd"] == LCD_MESSAGES["warning"]

    def test_starch_assessment_included(self):
        items = [self._item("Banku", 300, 200, 50, "Medium")]
        result = generate_recommendation(items)
        sa = result["starch_assessment"]
        assert "total_starch_volume_cm3" in sa
        assert "portion_category" in sa
        assert "ratio_to_reference" in sa

    def test_backward_compat_keys(self):
        """All keys from old assess_plate() must still be present."""
        items = [self._item("Salad", 500, 50, 15, "Low")]
        result = generate_recommendation(items)
        for key in ("ratios", "plate_balanced", "vegetables_low",
                     "alert_level", "overall_message"):
            assert key in result

    def test_assess_plate_alias(self):
        """assess_plate still works as an alias."""
        items = [self._item("Salad", 500, 50, 15, "Low")]
        result = assess_plate(items)
        assert "alert_level" in result


class TestEmptyPlate:
    def test_empty_items(self):
        result = generate_recommendation([])
        assert result["alert_level"] == "caution"
        assert result["starch_assessment"]["portion_category"] == "none"
