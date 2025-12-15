from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw" / "synthetic_disaster_events_2025.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"

# Columns (as in the dataset)
COL_EVENT_ID = "event_id"
COL_TYPE = "disaster_type"
COL_LOCATION = "location"
COL_LAT = "latitude"
COL_LON = "longitude"
COL_DATE = "date"
COL_SEVERITY = "severity_level"
COL_AFFECTED = "affected_population"
COL_LOSS = "estimated_economic_loss_usd"
COL_RESPONSE_H = "response_time_hours"
COL_AID = "aid_provided"
COL_INFRA = "infrastructure_damage_index"
COL_MAJOR = "is_major_disaster"

# Engineered time columns
COL_MONTH = "month"
COL_DOW = "day_of_week"

# Force components
COL_FORCE_HAZARD = "force_hazard_pressure"
COL_FORCE_EXPOSURE = "force_exposure_pressure"
COL_FORCE_LATENCY = "force_response_latency"
COL_FORCE_INFRA = "force_infra_fragility"
COL_FORCE_BUFFER = "force_buffer_capacity"

# Outputs
COL_INSTABILITY = "instability_index"
COL_ZONE = "early_warning_zone"
