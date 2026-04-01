import sqlite3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATABASE_PATH


def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                name                TEXT    NOT NULL,
                daily_carb_target_g REAL    DEFAULT 135.0,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS meals (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id              INTEGER NOT NULL,
                captured_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_image_path  TEXT,
                annotated_image_path TEXT,
                total_carbs_g        REAL DEFAULT 0,
                plate_assessment     TEXT,
                notes                TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS meal_items (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_id             INTEGER NOT NULL,
                food_name           TEXT    NOT NULL,
                confidence          REAL,
                portion_category    TEXT CHECK(portion_category IN ('small','appropriate','reduce','excessive')),
                estimated_volume_cm3 REAL,
                estimated_weight_g  REAL,
                carbs_g             REAL,
                glycemic_index      INTEGER,
                gi_classification   TEXT CHECK(gi_classification IN ('Low','Medium','High')),
                recommendation      TEXT,
                FOREIGN KEY (meal_id) REFERENCES meals(id) ON DELETE CASCADE
            );
        """)
        # Migrate: add plate_assessment column if missing
        cols = [r[1] for r in conn.execute("PRAGMA table_info(meals)").fetchall()]
        if 'plate_assessment' not in cols:
            conn.execute("ALTER TABLE meals ADD COLUMN plate_assessment TEXT")
    conn.close()
    print("Database initialised.")
