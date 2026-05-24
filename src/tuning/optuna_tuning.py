"""Optuna hyperparameter tuning with MLflow."""
import logging, os
import mlflow, mlflow.sklearn, optuna
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from src.experiments.sklearn_experiment import create_preprocessing_pipeline, generate_churn_data

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_objective(X_train, y_train, X_test, y_test, preprocessor):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        pipe = Pipeline([("pre", preprocessor), ("clf", GradientBoostingClassifier(**params, random_state=42))])
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
        trial.report(cv.mean(), 0)
        if trial.should_prune(): raise optuna.TrialPruned()
        pipe.fit(X_train, y_train)
        return f1_score(y_test, pipe.predict(X_test))
    return objective


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn"))
    n_trials = int(os.getenv("OPTUNA_N_TRIALS", "50"))
    data = generate_churn_data(5000)
    X, y = data.drop(columns=["churned"]), data["churned"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = create_preprocessing_pipeline()
    study = optuna.create_study(name="churn-hpo", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(create_objective(X_tr, y_tr, X_te, y_te, pre), n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best: F1={study.best_value:.4f}")
    with mlflow.start_run(run_name="tuned-gradient-boosting"):
        mlflow.set_tags({"model_type": "tuned-gb", "tuning": "optuna-tpe", "engineer": "gideon"})
        mlflow.log_params(study.best_params)
        pipe = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(**study.best_params, random_state=42))])
        pipe.fit(X_tr, y_tr)
        y_p = pipe.predict(X_te)
        f1 = f1_score(y_te, y_p); auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
        mlflow.log_metrics({"test_f1_score": f1, "test_auc_roc": auc})
        mlflow.sklearn.log_model(pipe, "model", signature=infer_signature(X_te, y_p),
            input_example=X_te.head(3), registered_model_name="customer-churn-model")
        logger.info(f"Registered: F1={f1:.4f}, AUC={auc:.4f}")


if __name__ == "__main__":
    main()
