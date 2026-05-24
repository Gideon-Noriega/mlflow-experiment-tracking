"""Tests for validation gates."""
import pytest
from src.registry.validation_gates import ValidationReport, ValidationResult, check_performance_metrics


@pytest.mark.unit
class TestValidationGates:
    def test_pass(self):
        assert all(r.passed for r in check_performance_metrics(
            {"test_accuracy": 0.9, "test_f1_score": 0.85, "test_auc_roc": 0.92}))
    def test_fail(self):
        assert all(not r.passed for r in check_performance_metrics(
            {"test_accuracy": 0.6, "test_f1_score": 0.5, "test_auc_roc": 0.55}))
    def test_custom_thresholds(self):
        assert check_performance_metrics({"test_f1_score": 0.7}, {"min_f1_score": 0.65})[0].passed
    def test_report(self):
        r = ValidationReport(model_uri="test")
        r.add_result(ValidationResult("t1", True, 0.9, 0.8)); assert r.overall_passed
        r.add_result(ValidationResult("t2", False, 0.5, 0.8)); assert not r.overall_passed
