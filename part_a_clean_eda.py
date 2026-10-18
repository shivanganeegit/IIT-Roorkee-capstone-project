import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# 1. Read input data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# 2. Clean shipments
clean, medians = clean_shipments(raw)

# 3. Build features
features = build_features(clean, hubs)

# 4. Add target column: is_delayed
features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)

# 5. Write modelling table
features.to_csv("data/modelling_table.csv", index=False)