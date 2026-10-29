
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Paths and pinned configuration
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

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

TARGET = "is_delayed"


# --------------------------------------------------
# 2. Load and validate the modelling table
# --------------------------------------------------

data_path = DATA_DIR / "modelling_table.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Cannot find {data_path}. Run Part A successfully first."
    )

data = pd.read_csv(data_path)

required_columns = FEATURES + [TARGET]
missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from modelling table: {missing_columns}"
    )

X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(data[TARGET], errors="coerce")

if X.isna().any().any():
    raise ValueError(
        "Feature columns contain missing or non-numeric values: "
        f"{X.columns[X.isna().any()].tolist()}"
    )

if y.isna().any():
    raise ValueError("Target contains missing or non-numeric values.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Feature data contains infinite values.")

if not np.isfinite(y.to_numpy(dtype=float)).all():
    raise ValueError("Target contains infinite values.")

if set(y.unique()) != {0, 1}:
    raise ValueError(
        f"Expected binary target values 0 and 1; found {sorted(y.unique())}"
    )

if len(data) != 1200:
    raise ValueError(
        f"Expected 1200 rows, but found {len(data)}."
    )

if not set(NOISE_COLUMNS).issubset(FEATURES):
    raise ValueError("NOISE_COLUMNS must be a subset of FEATURES.")


# --------------------------------------------------
# 3. Stratified train / validation / test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

actual_split = (
    len(X_train),
    len(X_val),
    len(X_test),
)

actual_delayed = (
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum()),
)

print("Split sizes:", actual_split)
print("Delayed counts:", actual_delayed)

if actual_split != (840, 180, 180):
    raise ValueError(f"Unexpected split sizes: {actual_split}")

if actual_delayed != (195, 42, 41):
    raise ValueError(
        "Unexpected stratified delayed counts: "
        f"{actual_delayed}. Check the Part A target and row order."
    )


# --------------------------------------------------
# 4. Metrics helper
# --------------------------------------------------

def calculate_classification_metrics(y_true, y_pred):
    """Return confusion-matrix cells and classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


# --------------------------------------------------
# 5. Blind baseline and three classifiers
# --------------------------------------------------

model_rows = []

# Predict every validation shipment as on time.
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_classification_metrics(
    y_val,
    blind_predictions,
)

# Assignment explicitly requires baseline precision to be 0.
blind_metrics["precision"] = 0.0

model_rows.append({
    "model": "blind_all_on_time",
    **blind_metrics,
})

# Logistic regression: scale inside the pipeline.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

model_rows.append({
    "model": "logistic_regression",
    **calculate_classification_metrics(y_val, logistic_predictions),
})

# Select decision-tree depth using validation F1.
tree_depths = [2, 3, 4, 5, 6, 8, 10]
tree_depth_results = []

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)

    metrics = calculate_classification_metrics(y_val, predictions)

    tree_depth_results.append({
        "depth": depth,
        "f1": metrics["f1"],
        "model": tree,
        "predictions": predictions,
    })

# Highest F1 wins; ties go to the smaller depth.
best_tree_result = sorted(
    tree_depth_results,
    key=lambda item: (-item["f1"], item["depth"]),
)[0]

best_tree_depth = best_tree_result["depth"]
best_tree_model = best_tree_result["model"]

model_rows.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **calculate_classification_metrics(
        y_val,
        best_tree_result["predictions"],
    ),
})

# Default random forest using all 17 features.
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

default_forest.fit(X_train, y_train)
default_forest_predictions = default_forest.predict(X_val)

model_rows.append({
    "model": "random_forest_default",
    **calculate_classification_metrics(
        y_val,
        default_forest_predictions,
    ),
})

models_df = pd.DataFrame(
    model_rows,
    columns=[
        "model",
        "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1",
    ],
)

models_df.to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 6. Feature scores and forest importance
# --------------------------------------------------

# Fit both importance calculations on training rows only.
forest_for_importance = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

forest_for_importance.fit(X_train, y_train)

f_selector = SelectKBest(
    score_func=f_classif,
    k="all",
)

f_selector.fit(X_train, y_train)

f_scores = f_selector.scores_

# Rank descending. NaN scores are ranked below numeric scores.
rankable_scores = np.nan_to_num(
    f_scores,
    nan=-np.inf,
    posinf=np.inf,
    neginf=-np.inf,
)

f_ranks = (
    pd.Series(rankable_scores)
    .rank(method="min", ascending=False)
    .astype(int)
    .to_numpy()
)

feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "f_rank": f_ranks,
    "rf_importance": forest_for_importance.feature_importances_,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
})

feature_scores_df = feature_scores_df.sort_values(
    ["f_rank", "feature"],
).reset_index(drop=True)

feature_scores_df.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False,
    float_format="%.6f",
)

noise_importance_share = float(
    feature_scores_df.loc[
        feature_scores_df["is_noise"],
        "rf_importance",
    ].sum()
)

top_seven_features = set(
    feature_scores_df.nsmallest(7, "f_rank")["feature"]
)

all_noise_outside_top_seven = all(
    feature_scores_df.loc[
        feature_scores_df["feature"] == column,
        "f_rank",
    ].iloc[0] >= 8
    for column in NOISE_COLUMNS
)


# --------------------------------------------------
# 7. SelectKBest experiment for k = 5, 7, 10, 12, 17
# --------------------------------------------------

selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    selectk_rows.append({
        "k": k,
        "val_f1": f1_score(
            y_val,
            predictions,
            zero_division=0,
        ),
    })

selectk_df = pd.DataFrame(selectk_rows)

selectk_df.to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 8. Fit the k=7 selector on training rows only
# --------------------------------------------------

selector_7 = SelectKBest(
    score_func=f_classif,
    k=7,
)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

selected_features = [
    feature
    for feature, selected in zip(FEATURES, selector_7.get_support())
    if selected
]

print("Selected seven features:", selected_features)

# Parameter grid has 3 x 3 x 3 = 27 combinations.
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}


# --------------------------------------------------
# 9. GridSearchCV and RandomizedSearchCV
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
)

grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.predict(X_val_7)
grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0,
)

grid_row = {
    "search": "grid_search",
    "candidates": 27,
    "cv_folds": 5,
    "total_fits": 27 * 5,
    "best_params": json.dumps(
        grid_search.best_params_,
        sort_keys=True,
    ),
    "best_cv_accuracy": float(grid_search.best_score_),
    "val_f1": float(grid_val_f1),
}

randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)

randomized_search.fit(X_train_7, y_train)

randomized_predictions = randomized_search.predict(X_val_7)
randomized_val_f1 = f1_score(
    y_val,
    randomized_predictions,
    zero_division=0,
)

randomized_row = {
    "search": "randomized_search",
    "candidates": 9,
    "cv_folds": 5,
    "total_fits": 9 * 5,
    "best_params": json.dumps(
        randomized_search.best_params_,
        sort_keys=True,
    ),
    "best_cv_accuracy": float(randomized_search.best_score_),
    "val_f1": float(randomized_val_f1),
}

# Untuned forest on the same seven selected features.
default_7_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    # max_depth=None and min_samples_leaf=1 are the defaults.
)

default_7_forest.fit(X_train_7, y_train)

default_7_predictions = default_7_forest.predict(X_val_7)

default_7_val_f1 = f1_score(
    y_val,
    default_7_predictions,
    zero_division=0,
)

default_7_row = {
    "search": "default_forest_same_7_features",
    "candidates": 1,
    "cv_folds": 0,
    "total_fits": 1,
    "best_params": "defaults",
    "best_cv_accuracy": np.nan,
    "val_f1": float(default_7_val_f1),
}

tuning_df = pd.DataFrame(
    [grid_row, randomized_row, default_7_row],
    columns=[
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1",
    ],
)

tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 10. Choose final configuration using validation F1
# --------------------------------------------------

# Tie priority: default forest, grid search, randomized search.
tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

winning_row = sorted(
    tuning_df.to_dict("records"),
    key=lambda row: (
        -float(row["val_f1"]),
        tie_priority[row["search"]],
    ),
)[0]

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_.copy()
elif chosen_from == "randomized_search":
    chosen_params = randomized_search.best_params_.copy()
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

# Final pipeline fits the selector and classifier on training rows.
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            **chosen_params,
        ),
    ),
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 11. Select alert threshold using validation recall
# --------------------------------------------------

# Positive-class probabilities on validation rows only.
positive_class_index = list(
    final_model.named_steps["model"].classes_
).index(1)

val_probabilities = final_model.predict_proba(X_val)[
    :, positive_class_index
]

threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
]

threshold_rows = []

for threshold in threshold_values:
    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_rows.append({
        "threshold": threshold,
        **calculate_classification_metrics(y_val, predictions),
    })

thresholds_df = pd.DataFrame(threshold_rows)

thresholds_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False,
    float_format="%.6f",
)

eligible_thresholds = thresholds_df.loc[
    thresholds_df["recall"] >= 0.80,
    "threshold",
]

if eligible_thresholds.empty:
    chosen_threshold = 0.1
else:
    chosen_threshold = float(eligible_thresholds.max())

chosen_threshold_row = thresholds_df.loc[
    thresholds_df["threshold"] == chosen_threshold
].iloc[0]


# --------------------------------------------------
# 12. Final test evaluation — once, at chosen threshold
# --------------------------------------------------

# Do not tune or change the model after viewing these results.
test_probabilities = final_model.predict_proba(X_test)[
    :, positive_class_index
]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_classification_metrics(
    y_test,
    test_predictions,
)

final_test_df = pd.DataFrame([
    {
        "threshold": chosen_threshold,
        **test_metrics,
    }
])

final_test_df.to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 13. Save final configuration
# --------------------------------------------------

final_config_df = pd.DataFrame([
    {
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": (
            "unlimited"
            if chosen_params.get("max_depth") is None
            else chosen_params["max_depth"]
        ),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": chosen_from,
    }
])

final_config_df.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False,
)


# --------------------------------------------------
# 14. Generate README section from actual results
# --------------------------------------------------

blind_row = models_df.loc[
    models_df["model"] == "blind_all_on_time"
].iloc[0]

tuned_f1_best = max(
    float(grid_row["val_f1"]),
    float(randomized_row["val_f1"]),
)

tuning_beat_default = tuned_f1_best > float(default_7_row["val_f1"])

if tuning_beat_default:
    tuning_interpretation = (
        f"The best tuned validation F1 was {tuned_f1_best:.4f}, "
        f"above the untuned forest's {default_7_val_f1:.4f}. "
        "This is evidence of a validation improvement, not proof that "
        "the tuned configuration will generalize better."
    )
elif np.isclose(tuned_f1_best, default_7_val_f1):
    tuning_interpretation = (
        f"The best tuned validation F1 ({tuned_f1_best:.4f}) was tied "
        f"with the untuned forest ({default_7_val_f1:.4f}). "
        "The extra tuning did not improve validation F1."
    )
else:
    tuning_interpretation = (
        f"The best tuned validation F1 ({tuned_f1_best:.4f}) was below "
        f"the untuned forest ({default_7_val_f1:.4f}). "
        "On this validation set, tuning did not improve F1."
    )

top7_text = ", ".join(
    feature_scores_df.nsmallest(7, "f_rank")["feature"].tolist()
)

threshold_precision = float(chosen_threshold_row["precision"])
threshold_recall = float(chosen_threshold_row["recall"])
threshold_f1 = float(chosen_threshold_row["f1"])

noise_statement = (
    "All five noise columns fall outside the top seven by F-score."
    if all_noise_outside_top_seven
    else "At least one noise column ranks in the top seven by F-score; "
         "inspect the feature-score report."
)

test_recall = float(test_metrics["recall"])
val_recall = threshold_recall
recall_difference = test_recall - val_recall

readme_text = f"""## Model results (classification)

- **Blind baseline:** Validation accuracy = {blind_row['accuracy']:.4f}. It predicts every parcel is on time, so it detects no delayed parcels (TP = 0, FN = {int(blind_row['FN'])}, recall = 0, F1 = 0). With approximately 23% delayed parcels, its apparently reasonable accuracy is misleading and it is not useful for delay alerts.
- **Feature signal and noise:** The five bookkeeping/noise columns together account for {noise_importance_share:.4f} ({100 * noise_importance_share:.2f}%) of the 17-feature forest importance. {noise_statement} The seven highest F-score features are: {top7_text}. See `part_c_feature_scores.csv` for each score, rank, and forest importance.
- **Number of selected features:** The `part_c_selectk.csv` report compares validation F1 for k = 5, 7, 10, 12, and 17. The selected configuration uses k = 7 as required. The table shows whether reducing the feature set improves F1 on this split; it does not guarantee the same ordering on new data.
- **Tuning:** {tuning_interpretation} Only 180 validation rows are available, so small F1 differences may reflect sampling variability. Cross-validation accuracy and validation F1 measure different things and should not be treated as interchangeable.
- **Alert threshold:** The chosen threshold is {chosen_threshold:.1f}, selected as the highest tested threshold with validation recall at least 0.80 (or 0.1 if none qualifies). At this threshold, validation precision = {threshold_precision:.4f}, recall = {threshold_recall:.4f}, and F1 = {threshold_f1:.4f}. Lowering the threshold generally catches more delayed parcels but can increase unnecessary alerts.
- **Final test evaluation:** At the same threshold of {chosen_threshold:.1f}, test precision = {test_metrics['precision']:.4f}, recall = {test_recall:.4f}, and F1 = {test_metrics['f1']:.4f}. Validation recall was {val_recall:.4f}; test recall differs by {recall_difference:+.4f}. The test set contains only {int(y_test.sum())} delayed parcels, so each additional missed or correctly detected delayed parcel changes recall by about {1 / int(y_test.sum()):.4f}. The test set was not used to select the model or threshold.
"""

(OUTPUT_DIR / "part_c_readme_section.md").write_text(
    readme_text,
    encoding="utf-8",
)


# --------------------------------------------------
# 15. Print summary and generated files
# --------------------------------------------------

print("\nPart C classification completed successfully.")

print("\nValidation model results:")
print(models_df.to_string(index=False))

print("\nNoise columns' combined forest importance:", f"{noise_importance_share:.4f}")
print("All noise columns outside top seven by F-score:", all_noise_outside_top_seven)

print("\nSelectKBest results:")
print(selectk_df.to_string(index=False))

print("\nTuning results:")
print(tuning_df.to_string(index=False))

print(f"\nWinning configuration source: {chosen_from}")
print(f"Chosen alert threshold: {chosen_threshold:.1f}")
print(
    "Validation precision / recall / F1:",
    f"{threshold_precision:.4f} / {threshold_recall:.4f} / {threshold_f1:.4f}",
)

print("\nFinal test metrics:")
print(final_test_df.to_string(index=False))

print("\nFiles written:")
for filename in [
    "part_c_models.csv",
    "part_c_feature_scores.csv",
    "part_c_selectk.csv",
    "part_c_tuning.csv",
    "part_c_thresholds.csv",
    "part_c_final_test.csv",
    "part_c_final_config.csv",
    "part_c_readme_section.md",
]:
    print(f"- {OUTPUT_DIR / filename}")
