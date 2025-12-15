from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import minmax01, safe_log1p


def compute_forces(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interpretable force components for each disaster event.

    Conventions:
    - Pressures are negative (destabilizing): in [-1, 0]
    - Buffers are positive (stabilizing): in [0, +1]

    Components:
    - Hazard Pressure: severity_level (higher severity => more destabilizing)
    - Exposure Pressure: affected_population (log-scaled)
    - Response Latency Pressure: response_time_hours (slower response => more destabilizing)
    - Infrastructure Fragility: infrastructure_damage_index (higher fragility => more destabilizing)
    - Buffer Capacity: aid_provided + fast response (higher buffer => stabilizing)

    Note: This does NOT claim causality. It is a structured, interpretable proxy model for stress.
    """ 
    df = df.copy()

    # Clean basic types
    for c in [config.COL_SEVERITY, config.COL_AFFECTED, config.COL_LOSS, config.COL_RESPONSE_H, config.COL_INFRA]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize hazards / exposure / latency / infra
    sev_n = minmax01(df[config.COL_SEVERITY])
    exp_n = minmax01(safe_log1p(df[config.COL_AFFECTED]))
    lat_n = minmax01(df[config.COL_RESPONSE_H])
    infra_n = minmax01(df[config.COL_INFRA])

    # Convert to destabilizing pressures in [-1, 0]
    df[config.COL_FORCE_HAZARD] = -sev_n
    df[config.COL_FORCE_EXPOSURE] = -exp_n
    df[config.COL_FORCE_LATENCY] = -lat_n
    df[config.COL_FORCE_INFRA] = -infra_n

    # Buffer capacity: aid and response speed
    aid = df.get(config.COL_AID, pd.Series(["Unknown"] * len(df), index=df.index))
    aid = aid.astype(str).str.strip().str.lower()
    aid_yes = aid.isin(["yes", "y", "true", "1"]).astype(float)

    # Faster response means stronger buffer: (1 - latency_norm)
    speed = (1.0 - lat_n).clip(0.0, 1.0)
    df[config.COL_FORCE_BUFFER] = (0.65 * aid_yes + 0.35 * speed).clip(0.0, 1.0)

    return df


def compute_instability(df: pd.DataFrame) -> pd.DataFrame:
    """Combine forces into a leading-signal Instability Index + early warning zones.

    Instability increases with:
    - magnitude of negative pressures
    - conflict/imbalance between pressures
    - weak buffers

    Output:
    - instability_index in [0, ~1.5]
    - early_warning_zone: Stable / Fragile / Unstable / Critical (quantile based)
    """ 
    df = compute_forces(df)

    pressures = df[
        [config.COL_FORCE_HAZARD, config.COL_FORCE_EXPOSURE, config.COL_FORCE_LATENCY, config.COL_FORCE_INFRA]
    ].fillna(0.0)

    neg_mag = (-pressures).mean(axis=1)  # 0..1
    imbalance = pressures.std(axis=1).fillna(0.0)  # 0..~
    weak_buffer = (1.0 - df[config.COL_FORCE_BUFFER]).clip(0.0, 1.0)

    # Nonlinear penalty when both hazard and exposure are high (compounding)
    compounding = ((-df[config.COL_FORCE_HAZARD]) * (-df[config.COL_FORCE_EXPOSURE])).clip(0.0, 1.0)

    df[config.COL_INSTABILITY] = (0.45 * neg_mag + 0.15 * imbalance + 0.25 * weak_buffer + 0.15 * compounding).clip(0.0, 1.5)

    q1 = df[config.COL_INSTABILITY].quantile(0.50)
    q2 = df[config.COL_INSTABILITY].quantile(0.75)
    q3 = df[config.COL_INSTABILITY].quantile(0.90)

    def zone(v: float) -> str:
        if v <= q1:
            return "🟢 Stable"
        if v <= q2:
            return "🟡 Fragile"
        if v <= q3:
            return "🟠 Unstable"
        return "🔴 Critical"

    df[config.COL_ZONE] = df[config.COL_INSTABILITY].astype(float).apply(zone)
    return df
