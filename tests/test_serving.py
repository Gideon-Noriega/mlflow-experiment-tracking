"""Tests for serving."""
import pytest
from src.serving.ab_router import ABRouter


@pytest.mark.unit
class TestABRouter:
    def test_init(self):
        r = ABRouter("test", "file:///tmp/x", 0.8, 0.2)
        assert r.champion_weight == 0.8 and not r.is_ready()
    def test_stats(self):
        r = ABRouter("test", "file:///tmp/x")
        r.log_prediction("champion", 1, 10); r.log_prediction("champion", 0, 8)
        r.log_prediction("challenger", 1, 12)
        s = r.get_stats()
        assert s["champion"]["predictions"] == 2 and s["challenger"]["predictions"] == 1
    def test_route_raises(self):
        with pytest.raises(RuntimeError): ABRouter("test", "file:///tmp/x").route()
