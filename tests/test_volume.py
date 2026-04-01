"""Tests for the volume estimation module."""
from pipeline.volume import classify_portion, PORTION_THRESHOLDS


class TestClassifyPortion:
    def test_small(self):
        assert classify_portion(50) == "small"

    def test_appropriate(self):
        assert classify_portion(200) == "appropriate"

    def test_reduce(self):
        assert classify_portion(300) == "reduce"

    def test_excessive(self):
        assert classify_portion(500) == "excessive"

    def test_zero_volume(self):
        assert classify_portion(0) == "small"

    def test_boundary_small_to_appropriate(self):
        # 220 * 0.50 = 110.0 — exactly at boundary
        assert classify_portion(110.0) == "small"
        assert classify_portion(111) == "appropriate"

    def test_boundary_appropriate_to_reduce(self):
        # 220 * 1.00 = 220 — exactly at boundary
        assert classify_portion(220) == "appropriate"
        assert classify_portion(221) == "reduce"
