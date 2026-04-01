"""
Shared fixtures for the FoodAI test suite.
"""
import os
import sys
import json
import tempfile
import shutil

import pytest
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect all filesystem paths to a temp directory for every test."""
    captures = tmp_path / "static" / "captures"
    captures.mkdir(parents=True)
    exports = tmp_path / "exports"
    exports.mkdir()
    db_path = str(tmp_path / "test.db")

    monkeypatch.setattr("config.DATABASE_PATH", db_path)
    monkeypatch.setattr("config.CAPTURES_DIR", str(captures))
    monkeypatch.setattr("config.EXPORTS_DIR", str(exports))

    # Also patch the already-imported references inside app/database modules
    monkeypatch.setattr("database.db.DATABASE_PATH", db_path)
    import app as app_module
    monkeypatch.setattr(app_module, "CAPTURES_DIR", str(captures))
    monkeypatch.setattr(app_module, "EXPORTS_DIR", str(exports))


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Create a fresh Flask test app with an isolated DB."""
    # Reset lazy singletons so tests don't leak state
    import app as app_module
    monkeypatch.setattr(app_module, "_segmenter", None)
    monkeypatch.setattr(app_module, "_depth_est", None)
    monkeypatch.setattr(app_module, "_nutrition_db", None)

    from database.db import init_db
    init_db()

    app_module.app.config["TESTING"] = True
    return app_module.app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal valid JPEG file and return its bytes."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (0, 200, 0)  # green square
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def seed_user(client):
    """Create a test user and return the user dict."""
    resp = client.post("/api/users", json={"name": "TestUser", "daily_carb_target_g": 130})
    return resp.get_json()
