from __future__ import annotations

import numpy as np
import pandas as pd


def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=s.index)
    mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
    if mx - mn == 0:
        return pd.Series(0.5, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0.0, 1.0)


def safe_log1p(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = s.clip(lower=0.0)
    return np.log1p(s)
