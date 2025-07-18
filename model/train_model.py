import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from lightfm import LightFM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import os
import json

# === Step 1: Load training data ===
data_path = "data_pipeline/processed/training_data.csv"
df = pd.read_csv(data_path)

# === Step 2: Encode user and item IDs ===
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["user_idx"] = user_encoder.fit_transform(df["visitorid"])
df["item_idx"] = item_encoder.fit_transform(df["itemid"])

# === Step 3: Create interaction matrix ===
df["interaction"] = 1
num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()

interaction_matrix = sp.coo_matrix(
    (df["interaction"], (df["user_idx"], df["item_idx"])),
    shape=(num_users, num_items)
)

# === Step 4: Load and process side features ===
user_df = pd.read_csv("data_pipeline/processed/user_features.csv")
item_df = pd.read_csv("data_pipeline/processed/item_features.csv")

# Filter only relevant users/items
user_df = user_df[user_df["visitorid"].isin(df["visitorid"])]
item_df = item_df[item_df["itemid"].isin(df["itemid"])]

# Re-encode user/item IDs
user_df["user_idx"] = user_encoder.transform(user_df["visitorid"])
item_df["item_idx"] = item_encoder.transform(item_df["itemid"])

# === Fix user features ===
# Rename columns to match LightFM expectations
user_df = user_df.rename(columns={
    "total_views": "user_total_views",
    "total_purchases": "user_total_purchases"
})
user_df["user_purchase_rate"] = (
    user_df["user_total_purchases"] / user_df["user_total_views"].replace(0, np.nan)
).fillna(0)

# Create user feature matrix
user_features_dict = user_df.set_index("user_idx")[
    ["user_total_views", "user_total_purchases", "user_purchase_rate"]
].to_dict("index")
user_vec = DictVectorizer(sparse=True)
user_features_matrix = user_vec.fit_transform(user_features_dict.values())

# === Fix item features ===
item_df = item_df.rename(columns={
    "total_views": "item_total_views",
    "total_purchases": "item_total_purchases"
})
item_df["item_purchase_rate"] = (
    item_df["item_total_purchases"] / item_df["item_total_views"].replace(0, np.nan)
).fillna(0)

# Create item feature matrix
item_features_dict = item_df.set_index("item_idx")[
    ["item_total_views", "item_total_purchases", "item_purchase_rate"]
].to_dict("index")
item_vec = DictVectorizer(sparse=True)
item_features_matrix = item_vec.fit_transform(item_features_dict.values())

# === Step 5: Train hybrid model ===
model = LightFM(loss='warp')

model.fit(
    interaction_matrix,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=10,
    num_threads=4
)

# === Step 6: Save artifacts ===
os.makedirs("model/artifacts", exist_ok=True)

# Save ID maps (convert keys to int for JSON safety)
user_id_map = {int(k): int(v) for k, v in zip(df["visitorid"], df["user_idx"])}
item_id_map = {int(k): int(v) for k, v in zip(df["itemid"], df["item_idx"])}

with open("model/artifacts/user_id_map.json", "w") as f:
    json.dump(user_id_map, f)
with open("model/artifacts/item_id_map.json", "w") as f:
    json.dump(item_id_map, f)

# Save encoders and model
with open("model/artifacts/lightfm_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/artifacts/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)
with open("model/artifacts/item_encoder.pkl", "wb") as f:
    pickle.dump(item_encoder, f)
with open("model/artifacts/user_vec.pkl", "wb") as f:
    pickle.dump(user_vec, f)
with open("model/artifacts/item_vec.pkl", "wb") as f:
    pickle.dump(item_vec, f)

print("✅ Hybrid LightFM model trained and saved to model/artifacts/")
