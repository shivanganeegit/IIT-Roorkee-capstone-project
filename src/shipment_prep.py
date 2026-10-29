import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    # 1. Drop exact duplicate rows
    df = raw.drop_duplicates()

    # 2. service_type: strip spaces and convert to title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode: replace missing with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: clean and convert to numeric
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # 6. parcel_weight_kg: treat >100 as missing
    # df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    df["parcel_weight_kg"] = df["parcel_weight_kg"].where(df["parcel_weight_kg"] <= 100)

    # 7. Fill missing values with medians
    if medians is None:
        medians = {
            "distance_km": df["distance_km"].median(),
            "parcel_weight_kg": df["parcel_weight_kg"].median(),
            "declared_value_inr": df["declared_value_inr"].median(),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians

def build_features(clean, hubs) :
    # 1. Left merge with hubs on hub_id
    df = clean.merge(hubs, on="hub_id", how="left")
    df["service_type"] = (
    df["service_type"]
    .astype("string")
    .str.strip()
    .str.title()
)
    # 2. promised_hours (integer from PROMISED_HOURS)
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    if df["promised_hours"].isna().any():
        unknown = df.loc[
            df["promised_hours"].isna(),
            "service_type",
        ].unique()
        raise ValueError(f"Unknown service types: {unknown}")

    df["promised_hours"] = df["promised_hours"].astype(int)

    # 3. is_cod (1 if payment_mode == "COD", else 0; Unknown gives 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # 4. is_weekend (1 if dispatch day is Saturday or Sunday)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)

    # 5. dispatch_hour (hour of dispatch timestamp)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # 6. log_declared_value (natural log of declared_value_inr)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df