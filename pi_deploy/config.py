import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'foodai.db')
CAPTURES_DIR  = os.path.join(BASE_DIR, 'static', 'captures')
EXPORTS_DIR   = os.path.join(BASE_DIR, 'exports')
MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'best_ncnn_model')
NUTRITION_DB_PATH = os.path.join(BASE_DIR, 'database', 'nutrition_db.json')

# Pipeline
CONFIDENCE_THRESHOLD        = 0.40
PLATE_DIAMETER_CM           = 25.0    # thesis FR-05 (v3.1)
PLATE_RADIUS_CM             = PLATE_DIAMETER_CM / 2.0
PLATE_AREA_CM2              = 3.14159 * PLATE_RADIUS_CM ** 2
EXPECTED_MAX_FOOD_HEIGHT_CM = 5.0     # assumed max for relative-mode scaling (not measured)
DAILY_CARB_TARGET_DEFAULT   = 135.0   # 45g x 3 meals (default, overridden per user)
STARCH_REFERENCE_VOLUME_CM3 = 220.0   # medium orange, dietician calibrated (v3.2)

# Camera
PREVIEW_RESOLUTION  = (640, 480)
CAPTURE_RESOLUTION  = (2304, 1296)
STREAM_JPEG_QUALITY = 70
