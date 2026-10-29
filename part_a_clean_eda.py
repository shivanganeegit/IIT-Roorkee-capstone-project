
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


# --------------------------------------------------
# 2. Load source data
# --------------------------------------------------

raw_path = DATA_DIR / "raw_shipments.csv"
hubs_path = DATA_DIR / "hubs.csv"

if not raw_path.exists():
    raise FileNotFoundError(f"Shipment data not found: {raw_path}")

if not hubs_path.exists():
    raise FileNotFoundError(f"Hub data not found: {hubs_path}")

raw = pd.read_csv(raw_path)
hubs = pd.read_csv(hubs_path)

raw_rows = len(raw)

print("Source data loaded.")
print(f"Raw shipment rows: {raw_rows}")
print(f"Hub rows: {len(hubs)}")


# --------------------------------------------------
# 3. Remove exact duplicate rows
# --------------------------------------------------

dedup = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(dedup)
rows_after_dedup = len(dedup)


# --------------------------------------------------
# 4. Capture data-quality checks before cleaning
# --------------------------------------------------

required_raw_columns = [
    "distance_km",
    "parcel_weight_kg",
    "declared_value",
    "payment_mode",
    "service_type",
]

missing_raw_columns = [
    column
    for column in required_raw_columns
    if column not in dedup.columns
]

if missing_raw_columns:
    raise ValueError(
        "Required columns missing from raw shipment data: "
        f"{missing_raw_columns}"
    )

# Convert to numeric for reliable checks, without modifying dedup.
weight_before = pd.to_numeric(
    dedup["parcel_weight_kg"],
    errors="coerce",
)

checks_before = {
    "missing_distance_km_before": int(
        dedup["distance_km"].isna().sum()
    ),
    "missing_parcel_weight_before": int(
        dedup["parcel_weight_kg"].isna().sum()
    ),
    "sentinel_weight_rows": int(
        (weight_before > 100).sum()
    ),
    "missing_declared_value_before": int(
        dedup["declared_value"].isna().sum()
    ),
    "missing_payment_mode_before": int(
        dedup["payment_mode"].isna().sum()
    ),
    "service_type_spellings_before": int(
        dedup["service_type"].nunique(dropna=True)
    ),
}


# --------------------------------------------------
# 5. Clean shipments and build features
# --------------------------------------------------

clean, medians = clean_shipments(dedup)

features = build_features(clean, hubs)

# build_features() must create promised_hours by mapping
# service_type using PROMISED_HOURS in src/shipment_prep.py.
# Do not overwrite promised_hours here.

if "promised_hours" not in features.columns:
    raise ValueError(
        "'promised_hours' was not created by build_features(). "
        "Check the service_type mapping in src/shipment_prep.py."
    )

if "delivery_hours" not in features.columns:
    raise ValueError(
        "'delivery_hours' is missing from the feature table. "
        "Check the raw data and clean_shipments()."
    )

# Validate required modelling columns.
required_columns = FEATURES + [
    "delivery_hours",
    "is_delayed",
]

# Add the target only after verifying delivery_hours and promised_hours.
features["is_delayed"] = (
    features["delivery_hours"] > features["promised_hours"]
).astype(int)

missing_columns = [
    column
    for column in required_columns
    if column not in features.columns
]

if missing_columns:
    raise ValueError(
        f"Required modelling columns are missing: {missing_columns}"
    )

# Do not silently treat missing delivery times as non-delayed.
target_inputs = [
    "delivery_hours",
    "promised_hours",
]

missing_target_inputs = features[target_inputs].isna().sum()
missing_target_inputs = missing_target_inputs[
    missing_target_inputs > 0
]

if not missing_target_inputs.empty:
    raise ValueError(
        "Missing values found in target inputs: "
        f"{missing_target_inputs.to_dict()}"
    )

null_count = int(
    features[required_columns].isna().sum().sum()
)

# Keep a separate modelling table containing only required columns.
modelling_table = features[required_columns].copy()

modelling_table.to_csv(
    DATA_DIR / "modelling_table.csv",
    index=False,
    float_format="%.4f",
)


# --------------------------------------------------
# 6. Save cleaning medians
# --------------------------------------------------

with open(
    MODELS_DIR / "cleaning_medians.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            key: float(value)
            for key, value in medians.items()
        },
        file,
        indent=4,
    )


# --------------------------------------------------
# 7. Calculate summary statistics and outliers
# --------------------------------------------------

hours = pd.to_numeric(
    features["delivery_hours"],
    errors="coerce",
).dropna()

if hours.empty:
    raise ValueError(
        "No valid delivery_hours values are available for analysis."
    )

q1 = hours.quantile(0.25)
q3 = hours.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = int(
    ((hours < lower_bound) | (hours > upper_bound)).sum()
)

declared_values = pd.to_numeric(
    clean["declared_value_inr"],
    errors="coerce",
)

positive_declared_values = declared_values[
    declared_values > 0
]

if positive_declared_values.empty:
    raise ValueError(
        "No positive declared_value_inr values are available "
        "to calculate logarithmic statistics."
    )

log_declared_values = np.log(positive_declared_values)

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": rows_after_dedup,
    **checks_before,
    "service_type_spellings_after": int(
        clean["service_type"].nunique(dropna=True)
    ),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(features),
    "nulls_in_features_and_target": null_count,
    "delayed_count": int(features["is_delayed"].sum()),
    "delay_rate": float(features["is_delayed"].mean()),
    "hours_q1": float(q1),
    "hours_q3": float(q3),
    "lower_bound": float(lower_bound),
    "upper_bound": float(upper_bound),
    "outlier_count": outlier_count,
    "declared_value_mean_inr": float(
        declared_values.mean()
    ),
    "declared_value_median_inr": float(
        declared_values.median()
    ),
    "log_declared_value_mean": float(
        log_declared_values.mean()
    ),
    "log_declared_value_median": float(
        log_declared_values.median()
    ),
}

checks_df = pd.DataFrame(
    [
        {
            "metric": metric,
            "value": round(float(value), 4),
        }
        for metric, value in checks.items()
    ]
)

checks_df.to_csv(
    OUTPUT_DIR / "part_a_checks.csv",
    index=False,
)


# --------------------------------------------------
# 8. Delay analysis by service, hub tier, and distance
# --------------------------------------------------

def summarize_group(data, dimension, column):
    """Summarize shipment counts, delay rates, and delivery times."""

    grouped = (
        data.groupby(column, dropna=False)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        )
        .reset_index()
        .rename(columns={column: "group"})
    )

    grouped.insert(0, "dimension", dimension)

    return grouped


group_frames = []

# Service type.
group_frames.append(
    summarize_group(
        features,
        "service_type",
        "service_type",
    )
)

# Hub tier.
group_frames.append(
    summarize_group(
        features,
        "hub_tier",
        "hub_tier",
    )
)

# Distance bands.
distance_bins = [0, 200, 400, 600, 2000]
distance_labels = [
    "0-200",
    "201-400",
    "401-600",
    "601+",
]

features["_distance_band"] = pd.cut(
    features["distance_km"],
    bins=distance_bins,
    labels=distance_labels,
    right=True,
    include_lowest=True,
)

distance_group = (
    features.groupby(
        "_distance_band",
        observed=False,
    )
    .agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    )
    .reset_index()
    .rename(columns={"_distance_band": "group"})
)

distance_group.insert(
    0,
    "dimension",
    "distance_band",
)

group_frames.append(distance_group)

delay_by_group = pd.concat(
    group_frames,
    ignore_index=True,
)

delay_by_group["group"] = delay_by_group["group"].astype(str)
delay_by_group["shipments"] = delay_by_group["shipments"].astype(int)
delay_by_group["delayed"] = delay_by_group["delayed"].astype(int)
delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)

delay_by_group.to_csv(
    OUTPUT_DIR / "part_a_delay_by_group.csv",
    index=False,
)


# --------------------------------------------------
# 9. Correlation matrix
# --------------------------------------------------

numeric_columns = features.select_dtypes(
    include=[np.number]
)

correlation_matrix = numeric_columns.corr()

correlation_matrix.round(4).to_csv(
    OUTPUT_DIR / "part_a_correlations.csv",
    index=True,
    index_label="feature",
)


# --------------------------------------------------
# 10. Print run summary
# --------------------------------------------------

print("\nPart A cleaning and EDA completed successfully.")
print(f"Raw rows: {raw_rows}")
print(f"Duplicates removed: {duplicate_rows_removed}")
print(f"Rows after deduplication: {rows_after_dedup}")
print(f"Rows after feature engineering: {len(features)}")
print(f"Delayed shipments: {checks['delayed_count']}")
print(f"Delay rate: {checks['delay_rate']:.4f}")
print(f"Nulls in required features and target: {null_count}")

print("\nFiles written:")
print(f"- {DATA_DIR / 'modelling_table.csv'}")
print(f"- {MODELS_DIR / 'cleaning_medians.json'}")
print(f"- {OUTPUT_DIR / 'part_a_checks.csv'}")
print(f"- {OUTPUT_DIR / 'part_a_delay_by_group.csv'}")
print(f"- {OUTPUT_DIR / 'part_a_correlations.csv'}")
