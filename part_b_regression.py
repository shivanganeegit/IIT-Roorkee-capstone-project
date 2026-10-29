
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# 1. Paths and pinned feature configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "delivery_hours"


# --------------------------------------------------
# 2. Load and validate modelling data
# --------------------------------------------------

data_path = DATA_DIR / "modelling_table.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Modelling table not found: {data_path}. "
        "Run Part A successfully first."
    )

data = pd.read_csv(data_path)

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in modelling_table.csv: "
        f"{missing_columns}"
    )

# Use the exact pinned feature order.
X = data[FEATURES].copy()
y = pd.to_numeric(data[TARGET], errors="coerce")

# Regression estimators require numeric, finite inputs.
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    bad_columns = X.columns[X.isna().any()].tolist()
    raise ValueError(
        f"Missing or non-numeric feature values in: {bad_columns}"
    )

if y.isna().any():
    raise ValueError(
        f"Target '{TARGET}' contains missing or non-numeric values."
    )

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Feature data contains infinite values.")

if not np.isfinite(y.to_numpy(dtype=float)).all():
    raise ValueError("Target data contains infinite values.")

if len(data) != 1200:
    raise ValueError(
        f"Expected 1200 modelling rows, but found {len(data)}. "
        "Check the Part A output."
    )


# --------------------------------------------------
# 3. Split into train, validation and test
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
)

if (len(X_train), len(X_val), len(X_test)) != (840, 180, 180):
    raise ValueError(
        "Unexpected split sizes: "
        f"train={len(X_train)}, val={len(X_val)}, "
        f"test={len(X_test)}"
    )


# --------------------------------------------------
# 4. Scaling diagnostic: fit on training rows only
# --------------------------------------------------

diagnostic_scaler = StandardScaler()

X_train_scaled = diagnostic_scaler.fit_transform(X_train)
X_test_scaled = diagnostic_scaler.transform(X_test)

train_scaled_max_abs_mean = float(
    np.abs(X_train_scaled.mean(axis=0)).max()
)

test_scaled_max_abs_mean = float(
    np.abs(X_test_scaled.mean(axis=0)).max()
)

checks = pd.DataFrame(
    [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
    ],
    columns=["metric", "value"],
)

checks.to_csv(
    OUTPUT_DIR / "part_b_checks.csv",
    index=False,
)

print("Split sizes:", len(X_train), len(X_val), len(X_test))
print(
    "Training scaled max absolute mean:",
    f"{train_scaled_max_abs_mean:.12f}",
)
print(
    "Test scaled max absolute mean:",
    f"{test_scaled_max_abs_mean:.6f}",
)


# --------------------------------------------------
# 5. Metric helper
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Return regression metrics in the required format."""

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
    }


metric_rows = []


def add_metric_row(model_name, split_name, y_true, y_pred):
    """Append one row to the final metrics report."""

    metric_rows.append(
        {
            "model": model_name,
            "split": split_name,
            **calculate_metrics(y_true, y_pred),
        }
    )


# --------------------------------------------------
# 6. Mean baseline
# --------------------------------------------------

baseline_value = float(y_train.mean())

baseline_val_pred = np.full(
    shape=len(y_val),
    fill_value=baseline_value,
)

add_metric_row(
    "mean_baseline",
    "val",
    y_val,
    baseline_val_pred,
)


# --------------------------------------------------
# 7. Decision-tree depth diagnostic
# --------------------------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_rows = []
tree_validation_scores = {}

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)

    train_rmse = float(
        np.sqrt(mean_squared_error(y_train, train_pred))
    )
    val_rmse = float(
        np.sqrt(mean_squared_error(y_val, val_pred))
    )

    depth_label = "unlimited" if depth is None else str(depth)

    tree_depth_rows.append(
        {
            "max_depth": depth_label,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        }
    )

    tree_validation_scores[depth_label] = val_rmse

tree_depth_df = pd.DataFrame(tree_depth_rows)

tree_depth_df.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False,
    float_format="%.6f",
)

# Select the lowest validation RMSE.
# On ties, prefer the smaller numerical depth.
# Unlimited is considered deeper than every finite depth.
depth_order = {
    str(depth): depth for depth in [1, 2, 3, 4, 6, 8, 12]
}
depth_order["unlimited"] = float("inf")

best_depth_label = min(
    tree_validation_scores,
    key=lambda label: (
        tree_validation_scores[label],
        depth_order[label],
    ),
)

best_depth = (
    None if best_depth_label == "unlimited"
    else int(best_depth_label)
)

print("Selected decision-tree depth:", best_depth_label)


# --------------------------------------------------
# 8. Fit four candidate models on training rows
# --------------------------------------------------

def make_candidates(selected_depth):
    """Create fresh, unfitted candidate estimators."""

    return {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=selected_depth,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
    }


candidates = make_candidates(best_depth)
fitted_models = {}
validation_rmse = {}

for model_name, model in candidates.items():
    model.fit(X_train, y_train)

    fitted_models[model_name] = model

    val_pred = model.predict(X_val)

    add_metric_row(
        model_name,
        "val",
        y_val,
        val_pred,
    )

    validation_rmse[model_name] = float(
        np.sqrt(mean_squared_error(y_val, val_pred))
    )


# --------------------------------------------------
# 9. Gradient-boosting learning-rate experiment
# --------------------------------------------------

learning_rate_rows = []

for learning_rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    learning_rate_rows.append(
        {
            "learning_rate": learning_rate,
            "val_rmse": float(
                np.sqrt(mean_squared_error(y_val, val_pred))
            ),
            "val_r2": float(r2_score(y_val, val_pred)),
        }
    )

learning_rate_df = pd.DataFrame(learning_rate_rows)

learning_rate_df.to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 10. Rank candidates and cross-validate the top two
# --------------------------------------------------

ranked_candidates = sorted(
    validation_rmse.items(),
    key=lambda item: item[1],
)

top_two_names = [
    model_name
    for model_name, _ in ranked_candidates[:2]
]

X_trainval = pd.concat(
    [X_train, X_val],
    axis=0,
)

y_trainval = pd.concat(
    [y_train, y_val],
    axis=0,
)

# Use the same KFold instance for both models.
shared_kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_rows = []

for model_name in top_two_names:
    cv_model = clone(candidates[model_name])

    scores = cross_val_score(
        cv_model,
        X_trainval,
        y_trainval,
        cv=shared_kfold,
    )

    cv_rows.append(
        {
            "model": model_name,
            "mean_r2": float(np.mean(scores)),
            "std_r2": float(np.std(scores)),
        }
    )

cv_df = pd.DataFrame(cv_rows)

cv_df.to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 11. Evaluate the final model on the test set once
# --------------------------------------------------

# The final model is selected by validation RMSE.
final_model_name = ranked_candidates[0][0]
final_model = fitted_models[final_model_name]

# This is the only prediction and scoring operation on X_test.
test_pred = final_model.predict(X_test)

add_metric_row(
    final_model_name,
    "test",
    y_test,
    test_pred,
)


# --------------------------------------------------
# 12. Save the required metrics CSV
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metric_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 13. Generate a README section from actual results
# --------------------------------------------------

baseline_row = metrics_df.loc[
    (metrics_df["model"] == "mean_baseline")
    & (metrics_df["split"] == "val")
].iloc[0]

final_test_row = metrics_df.loc[
    (metrics_df["model"] == final_model_name)
    & (metrics_df["split"] == "test")
].iloc[0]

best_val_rmse = ranked_candidates[0][1]

best_lr_row = learning_rate_df.loc[
    learning_rate_df["val_r2"].idxmax()
]

lr_rows = learning_rate_df.set_index("learning_rate")

depth1_row = tree_depth_df.loc[
    tree_depth_df["max_depth"] == "1"
].iloc[0]

best_tree_row = tree_depth_df.loc[
    tree_depth_df["max_depth"] == best_depth_label
].iloc[0]

unlimited_row = tree_depth_df.loc[
    tree_depth_df["max_depth"] == "unlimited"
].iloc[0]

best_tree_val_rmse = float(best_tree_row["val_rmse"])

high_bias_depths = []
high_variance_depths = []

# Transparent diagnostic rules:
# high bias: small train-validation gap, but training RMSE
# remains substantially above the best tree's validation RMSE.
# high variance: validation RMSE exceeds training RMSE by >= 5 hours.
for _, row in tree_depth_df.iterrows():
    gap = float(row["val_rmse"] - row["train_rmse"])
    label = str(row["max_depth"])

    if gap >= 5.0:
        high_variance_depths.append(label)
    elif (
        gap <= 2.0
        and float(row["train_rmse"]) > best_tree_val_rmse + 2.0
    ):
        high_bias_depths.append(label)

if high_bias_depths:
    bias_text = ", ".join(high_bias_depths)
else:
    bias_text = (
        "none under the stated diagnostic rule; "
        "inspect the depth table for smaller gaps and high errors"
    )

if high_variance_depths:
    variance_text = ", ".join(high_variance_depths)
else:
    variance_text = (
        "none with a validation-minus-training RMSE gap of at least 5 hours"
    )

lr_best = float(best_lr_row["learning_rate"])
lr_best_r2 = float(best_lr_row["val_r2"])

if 0.01 in lr_rows.index and 1.0 in lr_rows.index:
    slow_lr_r2 = float(lr_rows.loc[0.01, "val_r2"])
    aggressive_lr_r2 = float(lr_rows.loc[1.0, "val_r2"])

    slow_text = (
        f"At learning_rate=0.01, validation R² is {slow_lr_r2:.4f}. "
        "With 100 estimators, this smaller step may learn too slowly."
    )
    aggressive_text = (
        f"At learning_rate=1.0, validation R² is {aggressive_lr_r2:.4f}. "
        "Its larger updates may overshoot and generalize less well."
    )
else:
    slow_text = "Compare the 0.01 result in the learning-rate CSV."
    aggressive_text = "Compare the 1.0 result in the learning-rate CSV."

cv_sorted = cv_df.sort_values("mean_r2", ascending=False)
cv_best_name = str(cv_sorted.iloc[0]["model"])

val_best_name = final_model_name

if cv_best_name == val_best_name:
    cv_interpretation = (
        f"Cross-validation and validation both favor {val_best_name}. "
        "This agreement supports the selection."
    )
else:
    cv_interpretation = (
        f"Validation favors {val_best_name}, while cross-validation ranks "
        f"{cv_best_name} highest by mean R². The shipped model remains "
        f"{val_best_name} under the specified rule because the final model "
        "must be selected by validation RMSE. Review the fold mean and "
        "standard deviation before changing that decision."
    )

readme_section = f"""## Model results (regression)

- **Baseline:** Validation R² = {baseline_row['R2']:.4f}. The mean baseline predicts the training-target mean for every validation parcel. Its R² indicates how it performs relative to predicting the validation-target mean; a value near zero or below zero indicates little or no predictive value.
- **Validation winner:** `{final_model_name}`, with validation RMSE = {best_val_rmse:.4f} hours. RMSE measures the typical prediction error scale in hours and penalizes larger errors more strongly.
- **Tree-depth diagnostic:** Best depth = `{best_depth_label}`, with validation RMSE = {best_tree_val_rmse:.4f} hours. Depth 1 has train RMSE = {depth1_row['train_rmse']:.4f} and validation RMSE = {depth1_row['val_rmse']:.4f}. Depth 1's validation-minus-training gap is {depth1_row['val_rmse'] - depth1_row['train_rmse']:.4f} hours. Unlimited depth has train RMSE = {unlimited_row['train_rmse']:.4f} and validation RMSE = {unlimited_row['val_rmse']:.4f}, a gap of {unlimited_row['val_rmse'] - unlimited_row['train_rmse']:.4f} hours.
- **Bias and variance:** Under the diagnostic rules used here, high-bias depths: {bias_text}. High-variance depths (validation RMSE minus training RMSE at least 5 hours): {variance_text}. High bias is associated with errors remaining high on both training and validation data; high variance is associated with a large gap between them.
- **Learning rate:** The best of the tested learning rates is {lr_best:g}, with validation R² = {lr_best_r2:.4f}. {slow_text} {aggressive_text} See `outputs/part_b_gb_learning_rate.csv` for all three results.
- **Cross-validation:** {cv_interpretation} The two cross-validated models and their mean and standard deviation of R² are in `outputs/part_b_cv.csv`.
- **Final test evaluation:** The selected model `{final_model_name}` has test RMSE = {final_test_row['RMSE']:.4f} hours and test R² = {final_test_row['R2']:.4f}. The test set was reserved for this final evaluation.
"""

readme_output = OUTPUT_DIR / "part_b_readme_section.md"
readme_output.write_text(
    readme_section,
    encoding="utf-8",
)


# --------------------------------------------------
# 14. Print run summary
# --------------------------------------------------

print("\nPart B regression completed successfully.")

print("\nValidation ranking (lowest RMSE first):")
for rank, (name, rmse) in enumerate(ranked_candidates, start=1):
    print(f"{rank}. {name}: RMSE = {rmse:.4f}")

print(f"\nFinal model: {final_model_name}")
print(f"Test RMSE: {final_test_row['RMSE']:.4f}")
print(f"Test R2: {final_test_row['R2']:.4f}")

print("\nFiles written:")
for filename in [
    "part_b_checks.csv",
    "part_b_tree_depth.csv",
    "part_b_metrics.csv",
    "part_b_gb_learning_rate.csv",
    "part_b_cv.csv",
    "part_b_readme_section.md",
]:
    print(f"- {OUTPUT_DIR / filename}")
