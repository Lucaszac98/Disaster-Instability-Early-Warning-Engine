from __future__ import annotations

import pandas as pd

from . import config
from .forces import compute_instability


def load_raw() -> pd.DataFrame:
    if not config.DATA_RAW.exists():
        raise FileNotFoundError(f"Raw dataset not found: {config.DATA_RAW}")
    return pd.read_csv(config.DATA_RAW)


def load_processed() -> pd.DataFrame:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / "disaster_instability.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    df = load_raw()

    # Parse date -> month, day_of_week
    if config.COL_DATE in df.columns:
        dt = pd.to_datetime(df[config.COL_DATE], errors="coerce")
        df[config.COL_MONTH] = dt.dt.month
        df[config.COL_DOW] = dt.dt.dayofweek

    df = compute_instability(df)

    df.to_parquet(out_path, index=False)
    return df
