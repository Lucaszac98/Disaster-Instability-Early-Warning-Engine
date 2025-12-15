from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from . import config, data_prep
from .features import get_preprocessor


MODEL_PATH = config.MODELS_DIR / "risk_model_major_disaster.joblib"


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    model_path: Path


def train_risk_model(random_state: int = 42) -> TrainResult:
    """Train an interpretable ML model to estimate P(major disaster).

    Notes:
    - This is NOT a damage predictor.
    - It complements the instability index by producing a probabilistic escalation risk.
    """ 
    df = data_prep.load_processed()

    if config.COL_MAJOR not in df.columns:
        raise ValueError(f"Target column not found: {config.COL_MAJOR}")

    y = df[config.COL_MAJOR].astype(int)
    X = df.drop(columns=[config.COL_MAJOR])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pre, spec = get_preprocessor(df)

    # GradientBoostingClassifier is stable, interpretable-ish, and works well on tabular data.
    clf = GradientBoostingClassifier(random_state=random_state)

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return TrainResult(metrics=metrics, model_path=MODEL_PATH)


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first: python -m src.cli train"
        )
    return joblib.load(MODEL_PATH)
