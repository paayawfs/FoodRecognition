"""Tests for /api/infer — validates the bug fixes for field names,
plate_assessment persistence, and image URL conversion."""
import io
import json
import os
from unittest.mock import patch, MagicMock

import numpy as np


def _fake_pipeline_result():
    """Minimal valid return value for run_pipeline()."""
    return {
        "foods": [
            {
                "food_name": "Jollof_Rice",
                "display_name": "Jollof Rice",
                "confidence": 0.92,
                "portion_category": "appropriate",
                "volume_cm3": 180.0,
                "weight_g": 99.0,
                "area_cm2": 50.0,
                "carbs_g": 35.0,
                "calories": 170.0,
                "protein_g": 4.0,
                "fat_g": 3.5,
                "glycemic_index": 72,
                "gi_classification": "High",
                "recommendation": "Your Jollof Rice portion looks good.",
            }
        ],
        "totals": {"carbs_g": 35.0, "calories": 170.0, "protein_g": 4.0, "fat_g": 3.5},
        "plate_assessment": {
            "ratios": {"starch": 0.6, "protein": 0.2, "vegetable": 0.2},
            "plate_balanced": False,
            "vegetables_low": True,
            "alert_level": "caution",
            "overall_message": "Add more vegetables to your plate.",
            "detail_messages": ["Consider reducing your starch portion."],
            "starch_assessment": {
                "total_starch_volume_cm3": 180.0,
                "portion_category": "appropriate",
                "ratio_to_reference": 0.8,
                "message": "Your starch portion looks good.",
            },
            "gi_info": [{"food": "Jollof_Rice", "gi": 72, "gi_class": "High"}],
            "lcd": ("Add Vegetables", "or Reduce starch"),
        },
        "annotated_b64": "iVBORw0KGgo=",  # minimal base64 (will be decoded)
        "plate_method": "hough",
    }


class TestInferEndpoint:
    def _upload_image(self, client, sample_image):
        """Helper: upload an image and return the JSON response."""
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(sample_image), "test.jpg"),
        }, content_type="multipart/form-data")
        return resp.get_json()

    @patch("app.run_pipeline")
    def test_infer_with_filename(self, mock_pipeline, client, sample_image, seed_user):
        """Bug 1 fix: /api/infer accepts 'filename' (not 'capture_id')."""
        mock_pipeline.return_value = _fake_pipeline_result()
        upload = self._upload_image(client, sample_image)

        resp = client.post("/api/infer", json={
            "filename": upload["filename"],
            "user_id": seed_user["id"],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["meal_id"] is not None
        assert len(data["foods"]) == 1
        assert data["foods"][0]["food_name"] == "Jollof_Rice"
        assert data["plate_assessment"]["alert_level"] == "caution"

    def test_infer_missing_filename(self, client):
        resp = client.post("/api/infer", json={})
        assert resp.status_code == 400
        assert "filename required" in resp.get_json()["error"]

    def test_infer_file_not_found(self, client):
        resp = client.post("/api/infer", json={"filename": "nonexistent.jpg"})
        assert resp.status_code == 404

    @patch("app.run_pipeline")
    def test_plate_assessment_persisted(self, mock_pipeline, client, sample_image, seed_user):
        """Bug 4 fix: plate_assessment is stored in DB and returned by meal_detail."""
        mock_pipeline.return_value = _fake_pipeline_result()
        upload = self._upload_image(client, sample_image)

        infer_resp = client.post("/api/infer", json={
            "filename": upload["filename"],
            "user_id": seed_user["id"],
        })
        meal_id = infer_resp.get_json()["meal_id"]

        # Fetch via meal detail API
        detail_resp = client.get(f"/api/meals/detail/{meal_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.get_json()

        # plate_assessment should be deserialized from JSON
        pa = detail["meal"]["plate_assessment"]
        assert pa["alert_level"] == "caution"
        assert pa["ratios"]["starch"] == 0.6
        assert pa["overall_message"] == "Add more vegetables to your plate."

    @patch("app.run_pipeline")
    def test_image_urls_in_meal_detail(self, mock_pipeline, client, sample_image, seed_user):
        """Bug 3 fix: both original_url and annotated_url are proper URLs."""
        mock_pipeline.return_value = _fake_pipeline_result()
        upload = self._upload_image(client, sample_image)

        infer_resp = client.post("/api/infer", json={
            "filename": upload["filename"],
            "user_id": seed_user["id"],
        })
        meal_id = infer_resp.get_json()["meal_id"]

        detail = client.get(f"/api/meals/detail/{meal_id}").get_json()
        meal = detail["meal"]

        # Both should be URL paths, not filesystem paths
        assert meal["original_url"].startswith("/static/captures/")
        assert meal["annotated_url"].startswith("/static/captures/")
        # Should NOT contain backslashes or absolute paths
        assert ":\\" not in (meal["original_url"] or "")
        assert ":\\" not in (meal["annotated_url"] or "")

    @patch("app.run_pipeline")
    def test_meal_detail_nested_structure(self, mock_pipeline, client, sample_image, seed_user):
        """Bug 2 fix: /api/meals/detail returns { meal: {...}, items: [...] }."""
        mock_pipeline.return_value = _fake_pipeline_result()
        upload = self._upload_image(client, sample_image)

        infer_resp = client.post("/api/infer", json={
            "filename": upload["filename"],
            "user_id": seed_user["id"],
        })
        meal_id = infer_resp.get_json()["meal_id"]

        detail = client.get(f"/api/meals/detail/{meal_id}").get_json()

        # Verify nested structure
        assert "meal" in detail
        assert "items" in detail
        assert isinstance(detail["meal"], dict)
        assert isinstance(detail["items"], list)

        # Meal-level fields
        assert "total_carbs_g" in detail["meal"]
        assert "original_url" in detail["meal"]
        assert "annotated_url" in detail["meal"]
        assert "plate_assessment" in detail["meal"]

        # Items
        assert len(detail["items"]) == 1
        assert detail["items"][0]["food_name"] == "Jollof_Rice"
