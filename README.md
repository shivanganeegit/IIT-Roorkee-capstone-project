# How to run

This capstone builds a reproducible synthetic last-mile shipment pipeline: it cleans shipment records, explores delay patterns, evaluates delivery-time and late-delivery models, profiles hubs into operating segments, packages a classifier, and generates actions for new shipments.

The repository is organized around:

- `data/`: hub, raw shipment, new-shipment, and generated modelling CSVs.
- `src/shipment_prep.py`: shared cleaning and feature engineering used by training and prediction.
- `generate_data.py` and `part_*.py`: data generation and the four analysis stages.
- `package_model.py` and `predict_delay.py`: model packaging and scoring.
- `outputs/`: result tables and shipment-level actions.
- `models/`: the packaged model and the threshold, feature order, and imputation values it needs.

From a fresh clone, run these commands in Windows PowerShell from the repository directory. The supplied synthetic CSVs are already present. Run the first command only when you want to regenerate them; it replaces `data/hubs.csv`, `data/raw_shipments.csv`, and `data/new_shipments.csv`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Optional: regenerate the supplied synthetic input data.
python generate_data.py

python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

Run the seven pipeline commands in that order. Part A writes `data/modelling_table.csv`, `models/cleaning_medians.json`, and `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, and `outputs/part_a_correlations.csv`. Part B writes `outputs/part_b_checks.csv`, `outputs/part_b_tree_depth.csv`, `outputs/part_b_gb_learning_rate.csv`, `outputs/part_b_cv.csv`, `outputs/part_b_metrics.csv`, and `outputs/part_b_readme_section.md`. Part C writes `outputs/part_c_models.csv`, `outputs/part_c_feature_scores.csv`, `outputs/part_c_selectk.csv`, `outputs/part_c_tuning.csv`, `outputs/part_c_thresholds.csv`, `outputs/part_c_final_test.csv`, `outputs/part_c_final_config.csv`, and `outputs/part_c_readme_section.md`. Part D writes the hub profile, hub segments, segment profile, tier mismatches, and K-separation table under `outputs/`.

`package_model.py` uses the selected configuration in `outputs/part_c_final_config.csv` and regenerates `models/delay_model.pkl`, `models/delay_threshold.csv`, `models/feature_order.csv`, and `models/impute_values.csv`; it also writes `outputs/part_d_reload_check.csv`. `predict_delay.py` reads those artifacts and `data/new_shipments.csv`, then writes `outputs/new_shipment_actions.csv`.

The model is stored as a scikit-learn pickle. Pickle compatibility depends on the scikit-learn version used to create it; rebuild the model with `python package_model.py` in the target environment rather than assuming a pickle will load across incompatible versions. This repository's current pins were load-tested against the supplied model.

# Data findings

`outputs/part_a_checks.csv` reports 1,224 raw rows, 24 exact duplicate rows removed, and 1,200 rows after deduplication and hub merge. All 1,200 modelling rows have complete feature and target values after cleaning. The target definition is `delivery_hours > promised_hours`; 278 shipments are delayed, a measured delay rate of 0.2317 (23.17%).

| Group | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 0.0857 |
| Express | 243 | 119 | 0.4897 |
| Standard | 642 | 132 | 0.2056 |

Express has the highest observed service-type delay rate. Delay rates increase by hub tier and distance band in these outputs:

| Dimension | Group | Shipments | Delayed | Delay rate |
|---|---|---:|---:|---:|
| Hub tier | 1 | 400 | 16 | 0.0400 |
| Hub tier | 2 | 400 | 86 | 0.2150 |
| Hub tier | 3 | 400 | 176 | 0.4400 |
| Distance (km) | 0-200 | 274 | 6 | 0.0219 |
| Distance (km) | 201-400 | 563 | 77 | 0.1368 |
| Distance (km) | 401-600 | 275 | 120 | 0.4364 |
| Distance (km) | 601+ | 88 | 75 | 0.8523 |

The three numeric variables most correlated with `delivery_hours`, ranked by absolute Pearson correlation in `outputs/part_a_correlations.csv` and excluding `delivery_hours`, are:

| Rank | Numeric column | Correlation with `delivery_hours` |
|---:|---|---:|
| 1 | `promised_hours` | +0.8372 |
| 2 | `distance_km` | +0.4013 |
| 3 | `n_stops` | +0.3247 |

Declared value is first cleaned to numeric INR and missing values are median-imputed; feature engineering then uses its natural logarithm (`log_declared_value`) to reduce the effect of its right-skewed scale. The cleaned values have a measured mean of INR 1,533.0542 and median of INR 1,121.5000; their logged mean and median are 7.0277 and 7.0224 respectively (`outputs/part_a_checks.csv`).

# Model results

## Delivery-time regression

Part B uses 840 training, 180 validation, and 180 test rows (`outputs/part_b_checks.csv`). The mean baseline and four candidate regressors were assessed using the validation set. The model was selected by lowest validation RMSE; the test set was reserved for one final evaluation.

| Model | Validation MAE (h) | Validation MSE (h2) | Validation RMSE (h) | Validation R2 |
|---|---:|---:|---:|---:|
| Mean baseline | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| Linear regression | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| Decision tree (depth 6) | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| Random forest | 4.2505 | 29.8987 | 5.4680 | 0.9141 |
| Gradient boosting | 3.9166 | 24.1722 | 4.9165 | 0.9306 |

Linear regression wins validation RMSE by a small margin over gradient boosting (4.8912 versus 4.9165 hours), so it is the selected regression model. On the untouched test split its MAE is 3.5671 hours, MSE is 20.4624 hours squared, RMSE is 4.5235 hours, and R2 is 0.9493 (`outputs/part_b_metrics.csv`). The 5-fold cross-validation comparison of the two validation leaders ranks gradient boosting higher by mean R2 (0.9352, standard deviation 0.0044) than linear regression (0.9261, standard deviation 0.0048; `outputs/part_b_cv.csv`). The shipped choice nevertheless follows the documented validation-RMSE rule, not the cross-validation ranking.

The tree-depth diagnostic's lowest validation RMSE is at depth 6 (6.2796 hours); the unlimited tree has training RMSE 0.0000 and validation RMSE 7.7890 hours (`outputs/part_b_tree_depth.csv`). In the separate gradient-boosting learning-rate check, 0.1 gives the best validation R2 (0.9306), versus 0.7127 at 0.01 and 0.8429 at 1.0 (`outputs/part_b_gb_learning_rate.csv`). These are validation diagnostics, not test scores.

## Late-shipment classification

`outputs/part_c_models.csv` compares the following validation results on 180 rows, including 42 delayed shipments:

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind all-on-time baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision tree, depth 6 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Default random forest, all 17 features | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The final configuration is a random forest with `SelectKBest(k=7)`, 100 trees, unlimited maximum depth, and `min_samples_leaf=1` (`outputs/part_c_final_config.csv`). `outputs/part_c_feature_scores.csv` ranks distance, hub tier, hub capacity, number of stops, promised hours, rain, and sale-event flag as the seven highest F-score features; the five bookkeeping/noise columns rank eighth or lower. In the feature-count experiment, k=7 has the highest reported validation F1 (0.8333), compared with 0.6988 at k=5, 0.7857 at k=10, 0.8049 at k=12, and 0.7848 at k=17 (`outputs/part_c_selectk.csv`).

On the seven-feature validation model, the untuned default forest reports F1 0.8333. The grid search's best configuration (100 trees, depth 6, leaf size 1) reports 5-fold CV accuracy 0.8952 and validation F1 0.7750; the randomized search's best configuration (50 trees, depth 6, leaf size 1) reports CV accuracy 0.8905 and validation F1 0.7848 (`outputs/part_c_tuning.csv`). Since both tuned validation F1 scores are below the default forest's reported 0.8333, the untuned seven-feature forest is retained. The CV accuracy values and validation F1 values are different metrics and should not be compared as if interchangeable.

Threshold selection is performed on validation probabilities: choose the highest tested threshold whose recall is at least 0.80. In `outputs/part_c_thresholds.csv`, 0.5 has validation recall 0.8333 and 0.6 has recall 0.7619, so 0.5 is selected; its threshold-sweep validation precision is 0.7955 and F1 is 0.8140. The final test result is separate: at threshold 0.5, `outputs/part_c_final_test.csv` reports TN=125, FP=14, FN=10, TP=31, accuracy 0.8667, precision 0.6889, recall 0.7561, and F1 0.7209. The supplied pickle was independently loaded and scored on the same stratified test split; it reproduced that confusion matrix.

There is a validation-artifact discrepancy worth preserving: `part_c_selectk.csv` and `part_c_tuning.csv` report F1 0.8333 for k=7/default forest, while the 0.5 row in `part_c_thresholds.csv` reports F1 0.8140. Reconstructing validation predictions with the saved model also yields the threshold-sweep result. The README reports each table's value in its own context and does not treat them as one consistent measurement; regenerate Part C and reconcile the outputs before relying on the validation F1 comparison.

# Operations recommendation

Part D standardizes each hub profile using mean delivery hours, delay rate, mean distance, and mean stops, then forms three K-means groups and orders them by average delay rate. The measured profiles in `outputs/segment_profile.csv` are:

| Segment | Hubs | Mean delivery hours | Mean delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.1550 | 314.1763 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.4600 | 470.4757 | 84.3444 |

The Escalate hubs and their measured individual delay rates (`outputs/hub_profile.csv`) are: HUB13 Nagpur 0.4600, HUB15 Kochi 0.3600, HUB17 Patna 0.4600, HUB18 Ranchi 0.3800, HUB19 Guwahati 0.6200, HUB20 Raipur 0.4800, HUB21 Bhubaneswar 0.4000, HUB23 Siliguri 0.4400, and HUB24 Agartala 0.5400.

`outputs/tier_vs_segment_mismatch.csv` identifies these departures from the tier-to-segment expectation. Use observed segment assignment to prioritize review; a tier mismatch is a prompt to validate local conditions, not proof that the hub is mismanaged.

| Hub / city | Tier | Assigned segment | Operational implication |
|---|---:|---|---|
| HUB07 / Kolkata | 1 | Watch | Do not assume tier 1 means routine operations; review its elevated local delay profile and use Watch monitoring. |
| HUB13 / Nagpur | 2 | Escalate | Route flagged shipments to manager escalation despite the tier 2 expectation. |
| HUB15 / Kochi | 2 | Escalate | Route flagged shipments to manager escalation despite the tier 2 expectation. |
| HUB22 / Dehradun | 3 | Watch | Do not escalate based on tier alone; retain Watch actions while investigating the difference from the tier expectation. |

The alert threshold in `models/delay_threshold.csv` is 0.5; prediction flags probabilities at or above that value. On the final test set, precision 0.6889 means about 6.9 of every 10 flagged shipments were actually delayed. Recall 0.7561 means about 7.6 of every 10 delayed shipments were caught. These are final test proportions, not guarantees for future shipments.

The 48 rows in `outputs/new_shipment_actions.csv` produce 8 flagged shipments: 6 `Escalate to hub manager`, 2 `Proactive customer call`, and 40 `No action`. Thus 6 of these 48 predictions require hub-manager escalation. The implemented rules are: a flagged shipment in an Escalate hub gets `Escalate to hub manager`; any other flagged shipment gets `Proactive customer call`; an unflagged shipment gets `No action`.

`outputs/part_d_separation.csv` defines separation as mean between-cluster pairwise distance divided by mean within-cluster pairwise distance, using standardized hub profiles. The `cluster_sizes` are counts of hubs, sorted largest to smallest:

| K | Cluster sizes | Separation ratio |
|---:|---|---:|
| 2 | 15-9 | 2.8940 |
| 3 | 9-8-7 | 5.3928 |
| 4 | 8-7-5-4 | 6.3047 |
| 5 | 8-7-4-3-2 | 6.8866 |

Although separation rises at K=4 and K=5, those solutions create small groups of four hubs and of three/two hubs respectively from only 24 hubs. K=3 is retained to support the three requested operational playbooks with cluster sizes 9, 8, and 7, rather than introducing potentially fragile small-segment policies. `outputs/part_d_reload_check.csv` also records 180 test rows scored and a maximum probability difference of 0.0000 after model save/reload.

# Production plan

This is a synthetic capstone, not a production service. The following repository elements are useful readiness foundations; they do not establish production-grade availability or speed.

| Readiness characteristic | Code or artifact in this project |
|---|---|
| Accuracy | `outputs/part_b_metrics.csv` and `outputs/part_c_final_test.csv` keep regression and classifier test metrics distinct from validation metrics. |
| Reliability | Cleaning, feature, hub, and model-artifact checks fail explicitly for missing required inputs, unknown hubs, invalid feature sets, or invalid thresholds. `outputs/part_d_reload_check.csv` checks reload probability parity. |
| Reproducibility | Fixed random seeds in the generator and model scripts, checked row counts, and deterministic train/validation/test splits make the analysis reproducible for the supplied data. |
| Maintainability | Shared cleaning and feature construction are centralized in `src/shipment_prep.py`; `src.shipment_prep.FEATURES` defines model feature order. |
| Training/prediction consistency | `predict_delay.py` reuses `clean_shipments` and `build_features`, loads saved training medians and feature order, and `package_model.py` verifies saved-model probability parity. |
| Monitorability | Prediction output includes probability, flag, segment, and action; CSV reports capture group rates, test metrics, thresholds, and hub profiles. |

No latency, throughput, availability, intervention-cost, or live drift benchmark is included. Benchmark these against a real deployment workload before setting service-level claims.

The following are **recommended starting triggers**, not measured project results. Calibrate them against real operating volumes, service commitments, and intervention capacity before rollout.

| Monitoring metric | Recommended trigger | Recommended response |
|---|---|---|
| Completed-shipment delay rate | More than 5 percentage points above its matched reference rate for two consecutive weekly windows | Investigate service, hub, distance, and event mix; review staffing and lane capacity before changing the model. |
| Flagged-shipment share | More than 10 percentage points above the same-cohort reference for two consecutive days | Check input quality and probability calibration; confirm call and manager-escalation capacity before widening interventions. |
| Recall after outcomes mature | Below 0.70 over a rolling set of at least 100 delayed shipments | Review false negatives and threshold behavior; retrain or recalibrate only after a temporal validation and approval. |
| Critical-feature missing rate | Above 1% for any required model feature in a daily batch | Pause automated actions for the affected batch, repair the upstream feed, and rerun validation before resuming. |
| Probability-distribution drift | Population Stability Index (PSI) above 0.20 for two weekly windows against a versioned reference | Investigate feature and probability shifts; compare outcomes when available and start a retraining assessment if drift persists. |

Review monitoring weekly and retrain on a quarterly cadence as a starting point, or sooner after a material process, feature, or sustained drift change. Keep the currently deployed model as the champion; train a candidate using recent, outcome-matured data and validate chronologically, including separate sale and non-sale periods. Do not select thresholds on the final holdout. Before promotion, check feature completeness, precision/recall at the operational threshold, calibration, segment coverage, inference behavior, and parity of training versus scoring preprocessing. Bundle each release's pickle, feature order, imputation values, threshold, code version, and evaluation report. Promote atomically, retain the previous bundle, and roll back if a trigger is crossed or the validation gates fail.

The generator defines `sale_event_flag=1` from 15 through 28 September inclusive (timestamps from 15 September up to but not including 29 September). Monitor and validate this window against comparable sale windows, ideally matching campaign days and service/hub mix, rather than comparing it only with ordinary weeks. Track changes in sale-event share, Express/Standard/Economy mix, distance, stops, rain, and delay rate conditional on sale status; also compare probability distributions, flagged share, and matured precision/recall. A shift in these conditional feature distributions or in sale-period outcome/calibration rates may indicate drift or a changed promotion/operating process. The synthetic generator explicitly adds sale-related delivery-time effects, so these observed relationships are not evidence of real-world campaign effects.

# Limitations

- The shipment and hub data are synthetic. The results do not demonstrate performance on real Delhivery operations, routes, customer behavior, or service-level definitions.
- The final classification test set has 180 shipments and only 41 delayed cases (`TP + FN` in `outputs/part_c_final_test.csv`). One additional correctly detected or missed delayed case changes recall by about 1/41, so the estimate is sensitive to a small number of cases.
- The classifier threshold is selected from one 180-row validation split with 42 delayed shipments. Its chosen value and recall/precision trade-off may change on another split or production population.
- The Part C validation selection tables do not fully agree: k=7/default-forest F1 is 0.8333 in `part_c_selectk.csv` and `part_c_tuning.csv`, while the 0.5 threshold-sweep result in `part_c_thresholds.csv` is 0.8140 and is reproduced by validation scoring of the saved model. Reconcile these outputs by rerunning Part C before treating the validation comparison as a stable benchmark; the recorded final test result does match the saved model.
- Production feature distributions, delay prevalence, and operating conditions may drift. The 48-row new-shipment action output is an example batch, not a performance evaluation with known delivery outcomes.
- The repository has no measured operational cost or benefit for calls, escalations, missed delays, or capacity changes. The action rules therefore cannot be claimed to be cost-optimal.
- The generated examples cover a defined 15-28 September sale window. They cannot establish how real seasonal promotions, weather, or changing network conditions affect delay rates.
