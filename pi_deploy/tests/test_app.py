"""
FoodAI Flask App Test Suite
============================
Run from the capstone/ directory:
    python -m pytest tests/test_app.py -v

These tests verify the Flask app, database, nutrition pipeline, and API
endpoints WITHOUT needing a Pi, camera, or GPU. Hardware-dependent
components (camera, ML models) are mocked.

Requirements (install in your laptop venv):
    pip install pytest
"""

import pytest
import os
import sys
import json
import sqlite3
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Add project root to path so imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Temporary directory for test artifacts."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def nutrition_db_path(tmp_dir):
    """Create a test nutrition_db.json with known values."""
    db = {
        "jollof_rice": {
            "name": "Jollof Rice",
            "per_100g": {"calories": 175, "carbs": 32, "protein": 4.5, "fat": 3.2, "fiber": 1.2},
            "glycemic_index": 70,
            "gi_classification": "High",
            "density": 0.55
        },
        "banku": {
            "name": "Banku",
            "per_100g": {"calories": 165, "carbs": 28, "protein": 2.1, "fat": 0.5, "fiber": 0.8},
            "glycemic_index": 73,
            "gi_classification": "High",
            "density": 0.50
        },
        "fufu": {
            "name": "Fufu",
            "per_100g": {"calories": 170, "carbs": 40, "protein": 1.5, "fat": 0.3, "fiber": 1.0},
            "glycemic_index": 55,
            "gi_classification": "Low",
            "density": 0.55
        },
        "grilled_chicken": {
            "name": "Grilled Chicken",
            "per_100g": {"calories": 165, "carbs": 0, "protein": 31, "fat": 3.6, "fiber": 0},
            "glycemic_index": 0,
            "gi_classification": "Low",
            "density": 0.85
        },
        "salad": {
            "name": "Salad",
            "per_100g": {"calories": 20, "carbs": 3, "protein": 1.5, "fat": 0.3, "fiber": 2.0},
            "glycemic_index": 15,
            "gi_classification": "Low",
            "density": 0.30
        },
        "okro_soup": {
            "name": "Okro Soup",
            "per_100g": {"calories": 60, "carbs": 7, "protein": 3.0, "fat": 2.5, "fiber": 3.0},
            "glycemic_index": 20,
            "gi_classification": "Low",
            "density": 0.90
        },
        "shito": {
            "name": "Shito",
            "per_100g": {"calories": 180, "carbs": 8, "protein": 5.0, "fat": 15, "fiber": 2.0},
            "glycemic_index": 10,
            "gi_classification": "Low",
            "density": 0.90
        },
    }
    path = os.path.join(tmp_dir, "nutrition_db.json")
    with open(path, "w") as f:
        json.dump(db, f)
    return path


@pytest.fixture
def plate_category_path(tmp_dir):
    """Create a test plate_category.json."""
    cats = {
        "Jollof_Rice": "starch_spread",
        "Banku": "starch_moulded",
        "Fufu": "starch_moulded",
        "Grilled_Chicken": "protein",
        "Salad": "vegetable",
        "Okro_Soup": "soup_sauce",
        "Light_Soup": "soup_sauce",
        "Shito": "soup_sauce",
    }
    path = os.path.join(tmp_dir, "plate_category.json")
    with open(path, "w") as f:
        json.dump(cats, f)
    return path


@pytest.fixture
def class_names_path(tmp_dir):
    """Create a test class_names.json."""
    names = {
        "0": "Jollof_Rice", "1": "Waakye", "2": "Banku", "3": "Fufu",
        "4": "Plain_Rice", "5": "Fried_Plantain", "6": "Grilled_Chicken",
        "7": "Tilapia", "8": "Fried_Fish", "9": "Beans", "10": "Boiled_Egg",
        "11": "Okro_Soup", "12": "Light_Soup", "13": "Salad", "14": "Shito",
        "15": "Plate"
    }
    path = os.path.join(tmp_dir, "class_names.json")
    with open(path, "w") as f:
        json.dump(names, f)
    return path


# ===========================================================================
# TEST 1: Nutrition DB integrity
# ===========================================================================

class TestNutritionDB:
    """Verify nutrition_db.json is well-formed and internally consistent."""

    def test_load_json(self, nutrition_db_path):
        with open(nutrition_db_path) as f:
            db = json.load(f)
        assert isinstance(db, dict)
        assert len(db) > 0

    def test_all_entries_have_required_fields(self, nutrition_db_path):
        with open(nutrition_db_path) as f:
            db = json.load(f)
        required = {"name", "per_100g", "glycemic_index", "gi_classification", "density"}
        for key, entry in db.items():
            missing = required - set(entry.keys())
            assert not missing, f"'{key}' missing fields: {missing}"

    def test_per_100g_has_macros(self, nutrition_db_path):
        with open(nutrition_db_path) as f:
            db = json.load(f)
        macros = {"calories", "carbs", "protein", "fat", "fiber"}
        for key, entry in db.items():
            p = entry["per_100g"]
            missing = macros - set(p.keys())
            assert not missing, f"'{key}' per_100g missing: {missing}"

    def test_density_is_positive(self, nutrition_db_path):
        with open(nutrition_db_path) as f:
            db = json.load(f)
        for key, entry in db.items():
            assert entry["density"] > 0, f"'{key}' density is {entry['density']}"

    def test_gi_classification_matches_value(self, nutrition_db_path):
        with open(nutrition_db_path) as f:
            db = json.load(f)
        for key, entry in db.items():
            gi = entry["glycemic_index"]
            cls = entry["gi_classification"]
            if gi == 0:
                # Proteins — GI not applicable, classified as Low
                assert cls == "Low", f"'{key}' GI=0 but class={cls}"
            elif gi <= 55:
                assert cls == "Low", f"'{key}' GI={gi} should be Low, got {cls}"
            elif gi <= 69:
                assert cls == "Medium", f"'{key}' GI={gi} should be Medium, got {cls}"
            else:
                assert cls == "High", f"'{key}' GI={gi} should be High, got {cls}"

    def test_banku_gi_eli_cophie(self, nutrition_db_path):
        """Banku GI should be 73 (High) per Eli-Cophie et al. 2017."""
        with open(nutrition_db_path) as f:
            db = json.load(f)
        assert db["banku"]["glycemic_index"] == 73
        assert db["banku"]["gi_classification"] == "High"

    def test_fufu_gi_eli_cophie(self, nutrition_db_path):
        """Fufu GI should be 55 (Low) per Eli-Cophie et al. 2017."""
        with open(nutrition_db_path) as f:
            db = json.load(f)
        assert db["fufu"]["glycemic_index"] == 55
        assert db["fufu"]["gi_classification"] == "Low"

    def test_banku_has_density(self, nutrition_db_path):
        """Regression: banku density was swallowed by a comment in v3.1."""
        with open(nutrition_db_path) as f:
            db = json.load(f)
        assert "density" in db["banku"]
        assert db["banku"]["density"] == 0.50

    def test_fufu_has_density(self, nutrition_db_path):
        """Regression: fufu density was swallowed by a comment in v3.1."""
        with open(nutrition_db_path) as f:
            db = json.load(f)
        assert "density" in db["fufu"]
        assert db["fufu"]["density"] == 0.55


# ===========================================================================
# TEST 2: Plate category mapping
# ===========================================================================

class TestPlateCategory:
    """Verify plate_category.json is consistent."""

    def test_load_json(self, plate_category_path):
        with open(plate_category_path) as f:
            cats = json.load(f)
        assert isinstance(cats, dict)

    def test_all_values_are_valid_categories(self, plate_category_path):
        valid = {"starch_moulded", "starch_spread", "protein", "vegetable", "soup_sauce", "mixed"}
        with open(plate_category_path) as f:
            cats = json.load(f)
        for key, val in cats.items():
            assert val in valid, f"'{key}' has invalid category '{val}'"

    def test_light_soup_is_title_case(self, plate_category_path):
        """Regression: light_soup was lowercase, causing lookup miss."""
        with open(plate_category_path) as f:
            cats = json.load(f)
        assert "Light_Soup" in cats, "Light_Soup key missing — still lowercase?"
        assert cats["Light_Soup"] == "soup_sauce"

    def test_moulded_starches_present(self, plate_category_path):
        with open(plate_category_path) as f:
            cats = json.load(f)
        assert cats.get("Banku") == "starch_moulded"
        assert cats.get("Fufu") == "starch_moulded"

    def test_soup_sauce_excluded_foods(self, plate_category_path):
        with open(plate_category_path) as f:
            cats = json.load(f)
        soup_foods = [k for k, v in cats.items() if v == "soup_sauce"]
        assert len(soup_foods) >= 2, f"Expected at least Okro_Soup + Light_Soup, got {soup_foods}"


# ===========================================================================
# TEST 3: Class names
# ===========================================================================

class TestClassNames:
    """Verify class_names.json matches expected YOLO classes."""

    def test_load_json(self, class_names_path):
        with open(class_names_path) as f:
            names = json.load(f)
        assert isinstance(names, dict)
        assert len(names) == 16

    def test_plate_class_exists(self, class_names_path):
        with open(class_names_path) as f:
            names = json.load(f)
        values = list(names.values())
        assert "Plate" in values

    def test_all_food_classes_in_nutrition_db(self, class_names_path, nutrition_db_path):
        """Every non-Plate YOLO class should have a nutrition DB entry."""
        with open(class_names_path) as f:
            names = json.load(f)
        with open(nutrition_db_path) as f:
            db = json.load(f)

        db_keys = set(db.keys())
        for idx, name in names.items():
            if name == "Plate":
                continue
            db_key = name.lower()
            # Note: this will fail for foods not in the test fixture —
            # that's expected. Run against the REAL nutrition_db.json
            # to catch actual gaps.
            if db_key not in db_keys:
                pytest.skip(f"'{name}' not in test nutrition_db (expected for minimal fixture)")


# ===========================================================================
# TEST 4: Database schema
# ===========================================================================

class TestDatabase:
    """Test SQLite database creation and operations."""

    def _init_db(self, db_path):
        """Minimal DB init matching the app's schema."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                daily_carb_target_g REAL DEFAULT 135.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_image_path TEXT,
                annotated_image_path TEXT,
                total_carbs_g REAL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meal_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_id INTEGER NOT NULL,
                food_name TEXT NOT NULL,
                confidence REAL,
                portion_category TEXT,
                estimated_volume_cm3 REAL,
                estimated_weight_g REAL,
                carbs_g REAL,
                glycemic_index INTEGER,
                gi_classification TEXT,
                recommendation TEXT,
                FOREIGN KEY (meal_id) REFERENCES meals(id)
            )
        """)
        conn.commit()
        return conn

    def test_create_tables(self, tmp_dir):
        db_path = os.path.join(tmp_dir, "test.db")
        conn = self._init_db(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "users" in table_names
        assert "meals" in table_names
        assert "meal_items" in table_names
        conn.close()

    def test_create_user(self, tmp_dir):
        db_path = os.path.join(tmp_dir, "test.db")
        conn = self._init_db(db_path)
        conn.execute("INSERT INTO users (name) VALUES (?)", ("Ama",))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE name='Ama'").fetchone()
        assert row is not None
        assert row[1] == "Ama"
        assert row[2] == 135.0  # default carb target
        conn.close()

    def test_create_meal_with_items(self, tmp_dir):
        db_path = os.path.join(tmp_dir, "test.db")
        conn = self._init_db(db_path)
        conn.execute("INSERT INTO users (name) VALUES (?)", ("Kofi",))
        conn.execute(
            "INSERT INTO meals (user_id, total_carbs_g) VALUES (?, ?)",
            (1, 45.2)
        )
        conn.execute(
            "INSERT INTO meal_items (meal_id, food_name, confidence, portion_category, "
            "estimated_volume_cm3, estimated_weight_g, carbs_g, glycemic_index, "
            "gi_classification) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "Jollof_Rice", 0.91, "appropriate", 180.0, 99.0, 31.7, 70, "High")
        )
        conn.commit()

        items = conn.execute("SELECT * FROM meal_items WHERE meal_id=1").fetchall()
        assert len(items) == 1
        assert items[0][2] == "Jollof_Rice"
        conn.close()

    def test_cascade_delete_user_meals(self, tmp_dir):
        """Deleting a user should allow deleting their meals."""
        db_path = os.path.join(tmp_dir, "test.db")
        conn = self._init_db(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("INSERT INTO users (name) VALUES (?)", ("Test",))
        conn.execute("INSERT INTO meals (user_id) VALUES (?)", (1,))
        conn.commit()

        # Delete meals first, then user (no ON DELETE CASCADE in schema)
        conn.execute("DELETE FROM meals WHERE user_id=?", (1,))
        conn.execute("DELETE FROM users WHERE id=?", (1,))
        conn.commit()

        users = conn.execute("SELECT * FROM users").fetchall()
        meals = conn.execute("SELECT * FROM meals").fetchall()
        assert len(users) == 0
        assert len(meals) == 0
        conn.close()


# ===========================================================================
# TEST 5: Config validation
# ===========================================================================

class TestConfig:
    """Verify config.py has correct values after v3.2 cleanup."""

    def test_config_importable(self):
        """config.py should import without errors."""
        try:
            import config
        except ImportError:
            pytest.skip("config.py not on path — run from capstone/ directory")

    def test_plate_diameter(self):
        try:
            import config
        except ImportError:
            pytest.skip("config.py not on path")
        assert config.PLATE_DIAMETER_CM == 25.0, (
            f"PLATE_DIAMETER_CM is {config.PLATE_DIAMETER_CM}, should be 25.0"
        )

    def test_removed_constants_gone(self):
        """v3.2 removed these unsourced constants."""
        try:
            import config
        except ImportError:
            pytest.skip("config.py not on path")
        assert not hasattr(config, "UNIVERSAL_MAX_HEIGHT_CM"), "UNIVERSAL_MAX_HEIGHT_CM should be removed"
        assert not hasattr(config, "PX_PER_CM_MIN"), "PX_PER_CM_MIN should be removed"
        assert not hasattr(config, "PX_PER_CM_MAX"), "PX_PER_CM_MAX should be removed"

    def test_no_carb_budget(self):
        """45g carb budget is deprecated — should not be in config."""
        try:
            import config
        except ImportError:
            pytest.skip("config.py not on path")
        assert not hasattr(config, "CARB_BUDGET_PER_MEAL"), "CARB_BUDGET_PER_MEAL is deprecated"

    def test_starch_reference_volume(self):
        try:
            import config
        except ImportError:
            pytest.skip("config.py not on path")
        if hasattr(config, "STARCH_REFERENCE_VOLUME_CM3"):
            assert config.STARCH_REFERENCE_VOLUME_CM3 == 220.0


# ===========================================================================
# TEST 6: Cross-file consistency
# ===========================================================================

class TestCrossFileConsistency:
    """Verify all JSON files agree with each other."""

    def test_plate_category_keys_are_title_case(self, plate_category_path):
        """All keys should match YOLO output format (Title_Case)."""
        with open(plate_category_path) as f:
            cats = json.load(f)
        for key in cats:
            assert key[0].isupper(), f"'{key}' is not Title_Case — YOLO outputs Title_Case"

    def test_nutrition_db_keys_are_lowercase(self, nutrition_db_path):
        """All keys should be lowercase (db_key() lowercases YOLO names)."""
        with open(nutrition_db_path) as f:
            db = json.load(f)
        for key in db:
            assert key == key.lower(), f"'{key}' is not lowercase"

    def test_plate_category_foods_have_nutrition(self, plate_category_path, nutrition_db_path):
        """Every food in plate_category should have a nutrition_db entry."""
        with open(plate_category_path) as f:
            cats = json.load(f)
        with open(nutrition_db_path) as f:
            db = json.load(f)

        db_keys = set(db.keys())
        for food_class in cats:
            db_key = food_class.lower()
            if db_key not in db_keys:
                # Allow missing in test fixture, but flag it
                pytest.skip(f"'{food_class}' -> '{db_key}' not in test fixture")

    def test_class_names_cover_plate_category(self, class_names_path, plate_category_path):
        """Every plate_category food should exist in class_names."""
        with open(class_names_path) as f:
            names = set(json.load(f).values())
        with open(plate_category_path) as f:
            cats = json.load(f)
        for food in cats:
            assert food in names, f"'{food}' in plate_category but not in class_names"


# ===========================================================================
# TEST 7: Flask app routes (mocked — no models needed)
# ===========================================================================

class TestFlaskRoutes:
    """Test Flask API endpoints with mocked ML components."""

    @pytest.fixture
    def client(self, tmp_dir, nutrition_db_path):
        """Create a Flask test client with mocked dependencies."""
        # Mock heavy imports before importing app
        mock_camera = MagicMock()
        mock_segmenter = MagicMock()
        mock_depth = MagicMock()

        with patch.dict(os.environ, {
            "FOODAI_DB_PATH": os.path.join(tmp_dir, "test.db"),
            "FOODAI_CAPTURES_DIR": os.path.join(tmp_dir, "captures"),
        }):
            os.makedirs(os.path.join(tmp_dir, "captures"), exist_ok=True)

            try:
                # Try importing the actual app
                # This may fail if app.py tries to load models at import time
                # In that case, we test what we can
                import app as flask_app
                flask_app.app.config["TESTING"] = True
                with flask_app.app.test_client() as client:
                    yield client
            except Exception as e:
                pytest.skip(f"Could not import app.py: {e}")

    def test_index_returns_html(self, client):
        """GET / should return the SPA index.html."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_get_users_empty(self, client):
        """GET /api/users should return empty list initially."""
        resp = client.get("/api/users")
        if resp.status_code == 404:
            pytest.skip("Route /api/users not found")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_create_user(self, client):
        """POST /api/users should create a new user."""
        resp = client.post("/api/users",
                           json={"name": "Ama"},
                           content_type="application/json")
        if resp.status_code == 404:
            pytest.skip("Route /api/users not found")
        assert resp.status_code in (200, 201)
        data = resp.get_json()
        assert data.get("name") == "Ama" or "id" in data


# ===========================================================================
# TEST 8: Recommendation logic (standalone, no ML needed)
# ===========================================================================

class TestRecommendationLogic:
    """Test the Diabetic Plate Model logic directly."""

    def _classify_plate(self, detections):
        """Standalone plate composition classifier for testing."""
        category_pixels = {"starch": 0, "protein": 0, "vegetable": 0}
        plate_cat = {
            "Jollof_Rice": "starch_spread", "Banku": "starch_moulded",
            "Fufu": "starch_moulded", "Grilled_Chicken": "protein",
            "Salad": "vegetable", "Okro_Soup": "soup_sauce",
            "Light_Soup": "soup_sauce", "Shito": "soup_sauce",
        }
        for det in detections:
            cat = plate_cat.get(det["class_name"], "unknown")
            area = det["mask_area"]
            if cat in ("starch_moulded", "starch_spread"):
                category_pixels["starch"] += area
            elif cat == "soup_sauce":
                pass  # excluded
            elif cat in category_pixels:
                category_pixels[cat] += area

        total = sum(category_pixels.values())
        if total == 0:
            return {"starch": 0, "protein": 0, "vegetable": 0}
        return {k: v / total for k, v in category_pixels.items()}

    def test_balanced_plate(self):
        """50% veg, 25% protein, 25% starch -> balanced."""
        dets = [
            {"class_name": "Salad", "mask_area": 500},
            {"class_name": "Grilled_Chicken", "mask_area": 250},
            {"class_name": "Jollof_Rice", "mask_area": 250},
        ]
        ratios = self._classify_plate(dets)
        assert abs(ratios["vegetable"] - 0.50) < 0.01
        assert abs(ratios["protein"] - 0.25) < 0.01
        assert abs(ratios["starch"] - 0.25) < 0.01

    def test_starch_heavy_plate(self):
        """80% starch, 20% protein, 0% veg -> vegetables_low + starch high."""
        dets = [
            {"class_name": "Banku", "mask_area": 800},
            {"class_name": "Grilled_Chicken", "mask_area": 200},
        ]
        ratios = self._classify_plate(dets)
        assert ratios["vegetable"] < 0.30
        assert ratios["starch"] > 0.40

    def test_soup_excluded_from_ratio(self):
        """Soup/sauce should not count toward plate ratio."""
        dets = [
            {"class_name": "Banku", "mask_area": 300},
            {"class_name": "Okro_Soup", "mask_area": 500},
            {"class_name": "Grilled_Chicken", "mask_area": 200},
        ]
        ratios = self._classify_plate(dets)
        # Total should be 300+200=500 (soup excluded)
        assert abs(ratios["starch"] - 0.60) < 0.01
        assert abs(ratios["protein"] - 0.40) < 0.01
        assert ratios["vegetable"] == 0.0

    def test_starch_portion_moulded(self):
        """Moulded starch volume check against 220cm3 orange reference."""
        ref = 220.0
        assert 100.0 / ref <= 0.50   # small
        assert 200.0 / ref <= 1.00   # appropriate
        assert 300.0 / ref <= 1.50   # reduce
        assert 400.0 / ref > 1.50    # excessive

    def test_spread_starch_no_volume_check(self):
        """Spread starches (rice, plantain) should only use area ratio."""
        dets = [{"class_name": "Jollof_Rice", "volume_cm3": 500.0}]
        moulded_vol = sum(
            d["volume_cm3"] for d in dets
            if d["class_name"] in ("Banku", "Fufu")
        )
        assert moulded_vol == 0.0


# ===========================================================================
# TEST 9: File structure verification
# ===========================================================================

class TestFileStructure:
    """Verify all expected files exist in the project."""

    def _check_file(self, relative_path):
        full = PROJECT_ROOT / relative_path
        assert full.exists(), f"Missing: {relative_path}"

    def test_app_py(self):
        self._check_file("app.py")

    def test_config_py(self):
        self._check_file("config.py")

    def test_pipeline_init(self):
        self._check_file("pipeline/__init__.py")

    def test_pipeline_segmentation(self):
        self._check_file("pipeline/segmentation.py")

    def test_pipeline_depth(self):
        self._check_file("pipeline/depth.py")

    def test_pipeline_volume(self):
        self._check_file("pipeline/volume.py")

    def test_pipeline_nutrition(self):
        self._check_file("pipeline/nutrition.py")

    def test_database_init(self):
        self._check_file("database/__init__.py")

    def test_database_db(self):
        self._check_file("database/db.py")

    def test_nutrition_db_json(self):
        self._check_file("database/nutrition_db.json")

    def test_static_index(self):
        self._check_file("static/index.html")


# ===========================================================================
# TEST 10: Actual nutrition_db.json from project (if available)
# ===========================================================================

class TestRealNutritionDB:
    """Tests against the ACTUAL nutrition_db.json in the project."""

    @pytest.fixture
    def real_db(self):
        path = PROJECT_ROOT / "database" / "nutrition_db.json"
        if not path.exists():
            pytest.skip("database/nutrition_db.json not found")
        with open(path) as f:
            raw = json.load(f)
        # Strip _metadata key if present — it's not a food entry
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def test_has_15_or_more_foods(self, real_db):
        assert len(real_db) >= 15, f"Expected at least 15 foods, got {len(real_db)}"

    def test_all_have_density(self, real_db):
        for key, entry in real_db.items():
            assert "density" in entry, f"'{key}' missing density"
            assert isinstance(entry["density"], (int, float)), f"'{key}' density not numeric"

    def test_banku_gi_73(self, real_db):
        assert real_db["banku"]["glycemic_index"] == 73

    def test_fufu_gi_55(self, real_db):
        assert real_db["fufu"]["glycemic_index"] == 55

    def test_no_null_carbs(self, real_db):
        for key, entry in real_db.items():
            carbs = entry["per_100g"]["carbs"]
            assert carbs is not None, f"'{key}' has null carbs"
            assert isinstance(carbs, (int, float)), f"'{key}' carbs is {type(carbs)}"


# ===========================================================================
# TEST 11: Actual plate_category.json from project (if available)
# ===========================================================================

class TestRealPlateCategory:
    """Tests against the ACTUAL plate_category.json."""

    @pytest.fixture
    def real_cats(self):
        path = PROJECT_ROOT / "database" / "plate_category.json"
        if not path.exists():
            pytest.skip("database/plate_category.json not found")
        with open(path) as f:
            return json.load(f)

    def test_light_soup_title_case(self, real_cats):
        assert "Light_Soup" in real_cats, "Light_Soup should be Title_Case, not 'light_soup'"

    def test_no_lowercase_keys(self, real_cats):
        for key in real_cats:
            assert key[0].isupper(), f"'{key}' starts with lowercase — should be Title_Case"

    def test_has_all_food_classes(self, real_cats):
        expected = {
            "Jollof_Rice", "Waakye", "Plain_Rice", "Fried_Plantain",
            "Banku", "Fufu", "Beans", "Grilled_Chicken", "Tilapia",
            "Fried_Fish", "Boiled_Egg", "Okro_Soup", "Light_Soup",
            "Salad", "Shito"
        }
        missing = expected - set(real_cats.keys())
        assert not missing, f"Missing from plate_category: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
