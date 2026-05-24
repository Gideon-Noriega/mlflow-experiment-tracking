"""Model validation gates."""
import logging, time
from dataclasses import dataclass, field
import mlflow
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    name: str; passed: bool; actual_value: float; threshold: float; message: str = ""


@dataclass
class ValidationReport:
    model_uri: str; results: list = field(default_factory=list); overall_passed: bool = True
    def add_result(self, r):
        self.results.append(r)
        if not r.passed: self.overall_passed = False
    def summary(self):
        lines = [f"Report: {self.model_uri}"]
        for r in self.results:
            s = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{s}] {r.name}: {r.actual_value:.4f} (threshold: {r.threshold:.4f})")
        lines.append(f"Overall: {'PASSED' if self.overall_passed else 'FAILED'}")
        return "\n".join(lines)


DEFAULT_THRESHOLDS = {"min_accuracy": 0.80, "min_f1_score": 0.75, "min_auc_roc": 0.82,
                       "max_inference_time_ms": 50.0, "min_improvement_pct": 0.01}


def check_performance_metrics(metrics, thresholds=None):
    t = thresholds or DEFAULT_THRESHOLDS
    results = []
    for tk, mk in [("min_accuracy", "test_accuracy"), ("min_f1_score", "test_f1_score"),
                   ("min_auc_roc", "test_auc_roc")]:
        if tk not in t: continue
        tv, av = t[tk], metrics.get(mk, 0.0)
        results.append(ValidationResult(name=f"Perf: {mk}", passed=av>=tv, actual_value=av, threshold=tv))
    return results


def check_inference_latency(model_uri, n=100, max_ms=50.0):
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        sample = pd.DataFrame({"tenure_months": np.random.uniform(1,72,n),
            "monthly_charges": np.random.uniform(30,300,n),
            "total_charges": np.random.uniform(100,20000,n),
            "num_support_tickets": np.random.uniform(0,10,n),
            "days_since_last_login": np.random.uniform(0,60,n),
            "num_logins_last_30d": np.random.uniform(0,50,n),
            "contract_value": np.random.uniform(500,50000,n),
            "num_products": np.random.uniform(1,8,n),
            "contract_type": np.random.choice(["month-to-month","1yr","2yr"],n),
            "payment_method": np.random.choice(["cc","bt","inv"],n),
            "subscription_tier": np.random.choice(["basic","pro","ent"],n),
            "onboarding_completed": np.random.choice(["yes","no"],n)})
        model.predict(sample.head(5))
        s = time.time(); model.predict(sample)
        ms = (time.time()-s)*1000/n
        return ValidationResult("Latency", ms<=max_ms, ms, max_ms)
    except Exception as e:
        return ValidationResult("Latency", False, 999.0, max_ms, str(e))


def compare_against_champion(metrics, model_name, min_pct=0.01):
    try:
        client = mlflow.MlflowClient()
        champ = client.get_model_version_by_alias(model_name, "champion")
        cf1 = client.get_run(champ.run_id).data.metrics.get("test_f1_score", 0)
        nf1 = metrics.get("test_f1_score", 0)
        imp = (nf1-cf1)/max(cf1, 1e-10)
        return ValidationResult("Champion Comparison", imp>=min_pct, imp, min_pct)
    except Exception:
        return ValidationResult("Champion Comparison", True, 1.0, min_pct, "No champion")


def validate_model_quality(model_uri, metrics, thresholds=None, model_name="customer-churn-model"):
    report = ValidationReport(model_uri=model_uri)
    t = thresholds or DEFAULT_THRESHOLDS
    for r in check_performance_metrics(metrics, t): report.add_result(r)
    report.add_result(compare_against_champion(metrics, model_name))
    report.add_result(check_inference_latency(model_uri, max_ms=t.get("max_inference_time_ms", 50)))
    logger.info(report.summary())
    return report.overall_passed, report


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mn = os.getenv("MODEL_NAME", "customer-churn-model")
    client = mlflow.MlflowClient()
    vs = client.search_model_versions(f"name='{mn}'")
    if vs:
        latest = max(vs, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        validate_model_quality(f"models:/{mn}/{latest.version}", run.data.metrics)
