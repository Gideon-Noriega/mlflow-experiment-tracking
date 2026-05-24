"""Preprocessing pipeline as custom PyFunc."""
import mlflow.pyfunc
import numpy as np


class PreprocessingWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor=None, model=None):
        self.preprocessor = preprocessor
        self.model = model

    def load_context(self, context):
        import pickle
        with open(context.artifacts["preprocessor"], "rb") as f: self.preprocessor = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f: self.model = pickle.load(f)

    def predict(self, context, model_input, params=None):
        X = self.preprocessor.transform(model_input)
        if (params or {}).get("return_proba") and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
