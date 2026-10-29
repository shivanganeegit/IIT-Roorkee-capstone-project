from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_COLUMNS = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action",
]


def load_model_artifacts():
    """Load the model and its supporting package files."""
    model_path = MODEL_DIR / "delay_model.pkl"
    threshold_path = MODEL_DIR / "delay_threshold.csv"
    feature_order_path = MODEL_DIR / "feature_order.csv"
    medians_path = MODEL_DIR / "impute_values.csv"

    required_paths = [
        model_path,
        threshold_path,
        feature_order_path,
        medians_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                "Run package_model.py first."
            )

    model = joblib.load(model_path)

    threshold_df = pd.read_csv(threshold_path)
    if (
        "threshold" not in threshold_df.columns
        or len(threshold_df) != 1
    ):
        raise ValueError(
            "delay_threshold.csv must contain one threshold value."
        )

    threshold = float(threshold_df.loc[0, "threshold"])

    if not 0 < threshold < 1:
        raise ValueError("The saved threshold must be between 0 and 1.")

    feature_df = pd.read_csv(feature_order_path)
    if "feature" not in feature_df.columns:
        raise ValueError(
            "feature_order.csv must contain a 'feature' column."
        )

    feature_order = feature_df["feature"].tolist()

    if feature_order != FEATURES:
        raise ValueError(
            "Saved feature order does not match src.shipment_prep.FEATURES."
        )

    medians_df = pd.read_csv(medians_path)
    if not {"column", "median"}.issubset(medians_df.columns):
        raise ValueError(
            "impute_values.csv must contain 'column' and 'median'."
        )

    if medians_df["column"].duplicated().any():
        raise ValueError("Duplicate columns found in impute_values.csv.")

    medians = dict(
        zip(medians_df["column"], medians_df["median"])
    )

    expected_medians = {
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    }

    if set(medians) != expected_medians:
        raise ValueError(
            f"Expected imputation medians for {sorted(expected_medians)}; "
            f"found {sorted(medians)}."
        )

    return model, threshold, feature_order, medians


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, threshold, feature_order, medians = load_model_artifacts()

    new_path = DATA_DIR / "new_shipments.csv"
    hubs_path = DATA_DIR / "hubs.csv"
    segments_path = OUTPUT_DIR / "hub_segments.csv"

    for path in [new_path, hubs_path, segments_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    new_shipments = pd.read_csv(new_path)
    hubs = pd.read_csv(hubs_path)
    hub_segments = pd.read_csv(segments_path)

    if "shipment_id" not in new_shipments.columns:
        raise ValueError("new_shipments.csv must contain shipment_id.")

    if new_shipments["shipment_id"].isna().any():
        raise ValueError("new_shipments.csv contains missing shipment IDs.")

    if new_shipments["shipment_id"].duplicated().any():
        raise ValueError("new_shipments.csv contains duplicate shipment IDs.")

    expected_segment_columns = {"hub_id", "segment"}
    missing_segment_columns = (
        expected_segment_columns - set(hub_segments.columns)
    )
    if missing_segment_columns:
        raise ValueError(
            "hub_segments.csv is missing columns: "
            f"{sorted(missing_segment_columns)}"
        )

    # Preserve input order. The helper column also prevents the shared
    # cleaning function from removing distinct input records as duplicates.
    new_shipments = new_shipments.copy()
    new_shipments["_input_order"] = np.arange(len(new_shipments))

    # Apply the same cleaning logic as the training project, but use the
    # medians saved at packaging time rather than recalculating them.
    clean, _ = clean_shipments(
        new_shipments,
        medians=medians,
    )

    # Attach hub information and calculate the model features.
    featured = build_features(clean, hubs)

    # Validate hub mappings and segment assignments.
    if featured["hub_city"].isna().any():
        unknown_hubs = sorted(
            featured.loc[
                featured["hub_city"].isna(), "hub_id"
            ].dropna().unique()
        )
        raise ValueError(
            f"New shipments reference unknown hubs: {unknown_hubs}"
        )

    featured = featured.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    if featured["segment"].isna().any():
        unknown_segments = sorted(
            featured.loc[
                featured["segment"].isna(), "hub_id"
            ].dropna().unique()
        )
        raise ValueError(
            f"No segment assignment found for hubs: {unknown_segments}"
        )

    missing_features = set(feature_order) - set(featured.columns)
    if missing_features:
        raise ValueError(
            f"Engineered data is missing model features: "
            f"{sorted(missing_features)}"
        )

    X_new = featured[feature_order].copy()

    if X_new.isna().any().any():
        missing_counts = X_new.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            f"Model features contain missing values: {missing_counts}"
        )

    # Select the probability associated with class 1 (delayed).
    probabilities = model.predict_proba(X_new)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"The packaged model does not contain positive class 1: {classes}"
        )

    positive_class_index = classes.index(1)
    delay_probabilities = probabilities[:, positive_class_index]

    # Use unrounded probabilities for threshold decisions.
    flagged = delay_probabilities >= threshold

    actions = []
    for is_flagged, segment in zip(flagged, featured["segment"]):
        if not is_flagged:
            actions.append("No action")
        elif segment == "Escalate":
            actions.append("Escalate to hub manager")
        else:
            actions.append("Proactive customer call")

    # Round probabilities only for presentation in the output file.
    result = pd.DataFrame(
        {
            "shipment_id": featured["shipment_id"],
            "hub_id": featured["hub_id"],
            "segment": featured["segment"],
            "delay_probability": np.round(delay_probabilities, 4),
            "flagged": flagged,
            "action": actions,
            "_input_order": featured["_input_order"],
        }
    )

    result = (
        result.sort_values("_input_order")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )
    result = result[OUTPUT_COLUMNS]

    output_path = OUTPUT_DIR / "new_shipment_actions.csv"
    result.to_csv(output_path, index=False)

    # Basic output validation.
    if len(result) != len(new_shipments):
        raise RuntimeError(
            f"Expected {len(new_shipments)} output rows, got {len(result)}."
        )

    if result.isna().any().any():
        raise RuntimeError("The prediction output contains missing values.")

    if result["shipment_id"].tolist() != (
        pd.read_csv(new_path)["shipment_id"].tolist()
    ):
        raise RuntimeError(
            "Output shipment order does not match the input shipment order."
        )

    print("Prediction completed successfully.")
    print(f"Input shipments: {len(new_shipments)}")
    print(f"Flagged shipments: {int(result['flagged'].sum())}")
    print(f"Threshold: {threshold}")
    print(f"Output file: {output_path}")
    print("\nAction counts:")
    print(result["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
