from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import clean_shipments, build_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

SEGMENT_ORDER = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2,
}

EXPECTED_SEGMENT_BY_TIER = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}


def load_data():
    """Load, clean, and prepare shipment-level data."""
    raw_path = DATA_DIR / "raw_shipments.csv"
    hubs_path = DATA_DIR / "hubs.csv"

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    required_raw = {
        "shipment_id",
        "hub_id",
        "service_type",
        "distance_km",
        "n_stops",
        "delivery_hours",
    }
    missing = required_raw - set(raw.columns)
    if missing:
        raise ValueError(
            f"raw_shipments.csv is missing columns: {sorted(missing)}"
        )

    required_hubs = {"hub_id", "hub_city", "hub_tier"}
    missing = required_hubs - set(hubs.columns)
    if missing:
        raise ValueError(
            f"hubs.csv is missing columns: {sorted(missing)}"
        )

    # Validate required hub identifiers before joining.
    if hubs["hub_id"].duplicated().any():
        duplicates = hubs.loc[
            hubs["hub_id"].duplicated(keep=False), "hub_id"
        ].unique().tolist()
        raise ValueError(f"Duplicate hub IDs in hubs.csv: {duplicates}")

    # Use the shared cleaning logic, including median imputation.
    shipments, medians = clean_shipments(raw)

    # Reuse the shared feature engineering and hub join.
    shipments = build_features(shipments, hubs)

    if shipments["hub_city"].isna().any():
        unknown_hubs = sorted(
            shipments.loc[
                shipments["hub_city"].isna(), "hub_id"
            ].dropna().unique()
        )
        raise ValueError(
            f"Shipment hub IDs not found in hubs.csv: {unknown_hubs}"
        )

    shipments["delivery_hours"] = pd.to_numeric(
        shipments["delivery_hours"], errors="coerce"
    )
    shipments["n_stops"] = pd.to_numeric(
        shipments["n_stops"], errors="coerce"
    )

    if shipments["delivery_hours"].isna().any():
        raise ValueError("delivery_hours contains missing or invalid values.")

    if shipments["n_stops"].isna().any():
        raise ValueError("n_stops contains missing or invalid values.")

    shipments["is_delayed"] = (
        shipments["delivery_hours"] > shipments["promised_hours"]
    ).astype(int)

    print(f"Cleaned shipment rows: {len(shipments)}")
    print(f"Imputed distance median: {medians['distance_km']:.2f} km")

    return shipments, hubs


def build_hub_profile(df, hubs):
    """Aggregate cleaned shipment records into one row per hub."""
    profile = (
        df.groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "nunique"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
    )

    # Keep all known hubs and sort before scaling/clustering.
    profile = hubs[["hub_id"]].merge(
        profile,
        on="hub_id",
        how="left",
        validate="one_to_one",
    )
    profile = profile.sort_values("hub_id").reset_index(drop=True)

    if profile[PROFILE_FEATURES].isna().any().any():
        missing_hubs = profile.loc[
            profile[PROFILE_FEATURES].isna().any(axis=1), "hub_id"
        ].tolist()
        raise ValueError(
            "Cannot profile hubs with missing shipment metrics: "
            f"{missing_hubs}"
        )

    profile["shipments"] = profile["shipments"].astype(int)
    profile.to_csv(OUTPUT_DIR / "hub_profile.csv", index=False)

    return profile


def calculate_separation(X, labels):
    """
    Calculate the mean between-cluster pairwise distance divided by
    the mean within-cluster pairwise distance.

    X contains standardized hub-profile rows. Each unordered pair of
    distinct rows is counted once; self-distances are excluded.
    """
    labels = np.asarray(labels)
    distances = pairwise_distances(X, metric="euclidean")

    upper_i, upper_j = np.triu_indices(len(X), k=1)
    pair_distances = distances[upper_i, upper_j]

    different_cluster = labels[upper_i] != labels[upper_j]
    same_cluster = ~different_cluster

    between_distances = pair_distances[different_cluster]
    within_distances = pair_distances[same_cluster]

    if len(between_distances) == 0:
        return np.nan

    if len(within_distances) == 0:
        return np.inf

    mean_between = float(between_distances.mean())
    mean_within = float(within_distances.mean())

    if mean_within == 0:
        return np.inf if mean_between > 0 else np.nan

    return mean_between / mean_within


def cluster_profiles(profile):
    """Evaluate K=2..5 and create the required K=3 segmentation."""
    scaler = StandardScaler()
    X = scaler.fit_transform(profile[PROFILE_FEATURES])

    separation_rows = []
    models = {}

    for k in [2, 3, 4, 5]:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X)
        models[k] = (model, labels)

        sizes = sorted(
            pd.Series(labels).value_counts().tolist(),
            reverse=True,
        )

        separation_rows.append(
            {
                "k": k,
                "cluster_sizes": "-".join(map(str, sizes)),
                "separation_ratio": calculate_separation(X, labels),
            }
        )

    separation = pd.DataFrame(separation_rows)
    separation.to_csv(
        OUTPUT_DIR / "part_d_separation.csv",
        index=False,
    )

    # Use K=3 because the assignment specifies three playbooks.
    _, labels = models[3]
    clustered = profile.copy()
    clustered["cluster_id"] = labels

    mean_delay_by_cluster = (
        clustered.groupby("cluster_id")["delay_rate"].mean()
    )
    ordered_clusters = mean_delay_by_cluster.sort_values().index.tolist()

    cluster_to_segment = {
        ordered_clusters[0]: "Standard",
        ordered_clusters[1]: "Watch",
        ordered_clusters[2]: "Escalate",
    }
    clustered["segment"] = clustered["cluster_id"].map(
        cluster_to_segment
    )

    return clustered


def save_segment_outputs(clustered, hubs):
    """Write hub assignments, segment profiles, and tier mismatches."""
    hub_segments = clustered.merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one",
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id").reset_index(drop=True)

    hub_segments.to_csv(
        OUTPUT_DIR / "hub_segments.csv",
        index=False,
    )

    segment_profile = (
        clustered.groupby("segment", as_index=False)
        .agg(
            hubs=("hub_id", "nunique"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
    )

    segment_profile["_order"] = segment_profile["segment"].map(
        SEGMENT_ORDER
    )
    segment_profile = (
        segment_profile.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    segment_profile.to_csv(
        OUTPUT_DIR / "segment_profile.csv",
        index=False,
    )

    # Compare assigned operational segment with the expected segment
    # associated with each hub tier.
    hub_segments["expected_segment"] = (
        pd.to_numeric(hub_segments["hub_tier"], errors="coerce")
        .map(EXPECTED_SEGMENT_BY_TIER)
    )

    mismatches = hub_segments.loc[
        hub_segments["segment"] != hub_segments["expected_segment"],
        [
            "hub_id",
            "hub_city",
            "hub_tier",
            "segment",
            "expected_segment",
        ],
    ]

    mismatches.to_csv(
        OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
        index=False,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shipments, hubs = load_data()
    profile = build_hub_profile(shipments, hubs)
    clustered = cluster_profiles(profile)
    save_segment_outputs(clustered, hubs)

    print("\nPart D segmentation completed.")
    print(f"Hub profile rows: {len(profile)}")
    print(f"Hub segments: {OUTPUT_DIR / 'hub_segments.csv'}")
    print(f"Segment profile: {OUTPUT_DIR / 'segment_profile.csv'}")
    print(f"Separation results: {OUTPUT_DIR / 'part_d_separation.csv'}")
    print(
        "Tier mismatches: "
        f"{OUTPUT_DIR / 'tier_vs_segment_mismatch.csv'}"
    )


if __name__ == "__main__":
    main()
