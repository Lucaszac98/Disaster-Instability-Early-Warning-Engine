from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from . import config



def _to_str(x):
    """Convert categorical arrays/dataframes to string dtype (pickle-safe)."""
    return x.astype(str)

@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]


def get_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    numeric = [
        config.COL_SEVERITY,
        config.COL_AFFECTED,
        config.COL_RESPONSE_H,
        config.COL_INFRA,
        config.COL_LAT,
        config.COL_LON,
        config.COL_MONTH,
        config.COL_DOW,
        # force features can help model learn in the same language as the explanation layer
        config.COL_INSTABILITY,
        config.COL_FORCE_HAZARD,
        config.COL_FORCE_EXPOSURE,
        config.COL_FORCE_LATENCY,
        config.COL_FORCE_INFRA,
        config.COL_FORCE_BUFFER,
    ]
    numeric = [c for c in numeric if c in df.columns]

    categorical = [c for c in [config.COL_TYPE, config.COL_LOCATION, config.COL_AID] if c in df.columns]
    return FeatureSpec(numeric=numeric, categorical=categorical)


def _make_onehot():
    # sklearn >= 1.2 uses sparse_output, older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, FeatureSpec]:
    spec = get_feature_spec(df)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(_to_str, validate=False)),
            ("onehot", _make_onehot()),
        ]
    )


def get_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, FeatureSpec]:
    spec = get_feature_spec(df)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(_to_str, validate=False)),
            ("onehot", _make_onehot()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, spec.numeric),
            ("cat", cat_pipe, spec.categorical),
        ],
        remainder="drop",
    )
    return pre, spec
