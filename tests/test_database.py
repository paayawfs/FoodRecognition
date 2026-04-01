"""Tests for the database module: schema, migrations, and CRUD."""
import json
import sqlite3
from database.db import get_db, init_db


class TestInitDb:
    def test_tables_created(self, app):
        db = get_db()
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        db.close()
        assert "users" in tables
        assert "meals" in tables
        assert "meal_items" in tables

    def test_meals_has_plate_assessment_column(self, app):
        db = get_db()
        cols = [r[1] for r in db.execute("PRAGMA table_info(meals)").fetchall()]
        db.close()
        assert "plate_assessment" in cols

    def test_migration_idempotent(self, app):
        """Running init_db twice should not fail."""
        init_db()
        init_db()
        db = get_db()
        cols = [r[1] for r in db.execute("PRAGMA table_info(meals)").fetchall()]
        db.close()
        assert cols.count("plate_assessment") == 1


class TestUsersCrud:
    def test_create_user(self, client):
        resp = client.post("/api/users", json={"name": "Alice", "daily_carb_target_g": 120})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "Alice"
        assert data["daily_carb_target_g"] == 120.0
        assert "id" in data

    def test_create_user_missing_name(self, client):
        resp = client.post("/api/users", json={"name": ""})
        assert resp.status_code == 400

    def test_list_users(self, client, seed_user):
        resp = client.get("/api/users")
        assert resp.status_code == 200
        users = resp.get_json()
        assert any(u["name"] == "TestUser" for u in users)

    def test_get_user(self, client, seed_user):
        uid = seed_user["id"]
        resp = client.get(f"/api/users/{uid}")
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "TestUser"

    def test_get_user_not_found(self, client):
        resp = client.get("/api/users/9999")
        assert resp.status_code == 404

    def test_update_user(self, client, seed_user):
        uid = seed_user["id"]
        resp = client.put(f"/api/users/{uid}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "Updated"

    def test_delete_user(self, client, seed_user):
        uid = seed_user["id"]
        resp = client.delete(f"/api/users/{uid}")
        assert resp.status_code == 204
        resp2 = client.get(f"/api/users/{uid}")
        assert resp2.status_code == 404
