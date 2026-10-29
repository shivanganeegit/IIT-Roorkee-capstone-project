from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

TARGET = "is_delayed"
RANDOM_STATE = 42


def load_final_config():
    """Read the selected Part C model configuration."""
    path = OUTPUT_DIR / "part_c_final_config.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run part_c_classification.py first."
        )

    config_df = pd.read_csv(path)

    required = {
        "selector_k",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "threshold",
        "chosen_from",
    }
    missing = required - set(config_df.columns)

    if missing:
        raise ValueError(
            f"Part C config is missing columns: {sorted(missing)}"
        )

    if len(config_df) != 1:
        raise ValueError(
            "part_c_final_config.csv must contain exactly one row."
        )

    config = config_df.iloc[0]

    max_depth_value = config["max_depth"]
    if pd.isna(max_depth_value) or str(max_depth_value).strip().lower() in {
        "unlimited",
        "none",
        "",
    }:
        max_depth = None
    else:
        max_depth = int(max_depth_value)

    result = {
        "selector_k": int(config["selector_k"]),
        "n_estimators": int(config["n_estimators"]),
        "max_depth": max_depth,
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "threshold": float(config["threshold"]),
        "chosen_from": str(config["chosen_from"]),
    }

    if not 1 <= result["selector_k"] <= len(FEATURES):
        raise ValueError(
            f"selector_k must be between 1 and {len(FEATURES)}."
        )

    if not 0 < result["threshold"] < 1:
        raise ValueError("The selected threshold must be between 0 and 1.")

    return result


def make_pipeline(config):
    """Rebuild the selected Part C classification pipeline."""
    return Pipeline(
        steps=[
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=config["selector_k"],
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def reconstruct_split():
    """Recreate the Part C stratified train/validation/test split."""
    path = DATA_DIR / "modelling_table.csv"
    df = pd.read_csv(path)

    required = set(FEATURES) | {TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"modelling_table.csv is missing columns: {sorted(missing)}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # First split: 70% training and 30% temporary data.
    X_train, X_temporary, y_train, y_temporary = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split: divide the temporary 30% equally into validation
    # and test sets. The test set is the second returned partition.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temporary,
        y_temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temporary,
    )

    print(
        "Split sizes:",
        f"train={len(X_train)},",
        f"validation={len(X_validation)},",
        f"test={len(X_test)}",
    )

    if (len(X_train), len(X_validation), len(X_test)) != (840, 180, 180):
        raise ValueError(
            "Unexpected split sizes. Expected 840/180/180; "
            "check modelling_table.csv and Part C split settings."
        )

    return X_train, y_train, X_test, y_test


def save_imputation_values():
    """Save the three raw-data medians used by shared cleaning."""
    raw_path = DATA_DIR / "raw_shipments.csv"
    raw = pd.read_csv(raw_path)

    _, medians = clean_shipments(raw)

    median_df = pd.DataFrame(
        [
            {"column": column, "median": float(value)}
            for column, value in medians.items()
        ],
        columns=["column", "median"],
    )

    expected_columns = {
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    }

    if set(median_df["column"]) != expected_columns:
        raise ValueError(
            "Unexpected imputation columns: "
            f"{median_df['column'].tolist()}"
        )

    median_df.to_csv(MODEL_DIR / "impute_values.csv", index=False)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_final_config()
    X_train, y_train, X_test, y_test = reconstruct_split()

    # Train the selected model on training rows only.
    model = make_pipeline(config)
    model.fit(X_train, y_train)

    # Capture test probabilities before saving and reloading.
    original_probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        original_probabilities >= config["threshold"]
    ).astype(int)

    print("\nSelected configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nTest-set metrics at the selected threshold:")
    print(f"  Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")

    # Save the fitted model.
    model_path = MODEL_DIR / "delay_model.pkl"
    joblib.dump(model, model_path)

    # Save the threshold in a one-column CSV.
    pd.DataFrame(
        {"threshold": [config["threshold"]]}
    ).to_csv(
        MODEL_DIR / "delay_threshold.csv",
        index=False,
    )

    # Save the exact input feature order.
    pd.DataFrame(
        {"feature": FEATURES}
    ).to_csv(
        MODEL_DIR / "feature_order.csv",
        index=False,
    )

    # Save the same raw-data imputation medians used by cleaning.
    save_imputation_values()

    # Reload the persisted artifact and verify its probabilities.
    reloaded_model = joblib.load(model_path)
    reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    max_difference = float(
        np.max(np.abs(original_probabilities - reloaded_probabilities))
    )

    reload_check = pd.DataFrame(
        [
            {"metric": "test_rows_scored", "value": len(X_test)},
            {
                "metric": "reload_max_abs_probability_diff",
                "value": max_difference,
            },
        ]
    )

    reload_check.to_csv(
        OUTPUT_DIR / "part_d_reload_check.csv",
        index=False,
    )

    print("\nModel package created:")
    print(f"  {model_path}")
    print(f"  {MODEL_DIR / 'delay_threshold.csv'}")
    print(f"  {MODEL_DIR / 'feature_order.csv'}")
    print(f"  {MODEL_DIR / 'impute_values.csv'}")
    print(f"  {OUTPUT_DIR / 'part_d_reload_check.csv'}")
    print(f"\nTest rows scored: {len(X_test)}")
    print(f"Maximum probability difference after reload: {max_difference:.12g}")

    if max_difference > 1e-9:
        raise RuntimeError(
            "Reload verification failed: probability difference exceeds 1e-9."
        )

    print("Reload verification passed.")


if __name__ == "__main__":
    main()
