"""Tests for the /api/upload endpoint (Bug 1 & 3 fix validation)."""
import io
import os


class TestUploadEndpoint:
    def test_upload_valid_jpeg(self, client, sample_image):
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(sample_image), "meal.jpg"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "filename" in data
        assert data["filename"].startswith("upload_")
        assert data["filename"].endswith(".jpg")
        assert data["url"].startswith("/static/captures/")

    def test_upload_valid_png(self, client):
        import cv2
        import numpy as np
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(buf.tobytes()), "meal.png"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["filename"].endswith(".png")

    def test_upload_no_file(self, client):
        resp = client.post("/api/upload", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "No image file" in resp.get_json()["error"]

    def test_upload_empty_filename(self, client):
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(b""), ""),
        }, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_unsupported_format(self, client):
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(b"fake"), "document.pdf"),
        }, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Unsupported" in resp.get_json()["error"]

    def test_uploaded_file_exists_on_disk(self, client, sample_image, tmp_path):
        resp = client.post("/api/upload", data={
            "image": (io.BytesIO(sample_image), "test.jpg"),
        }, content_type="multipart/form-data")
        data = resp.get_json()
        import config
        filepath = os.path.join(config.CAPTURES_DIR, data["filename"])
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
