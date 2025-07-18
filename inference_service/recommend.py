import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from lightfm import LightFM
from sklearn.feature_extraction import DictVectorizer
import os

# === Load model artifacts ===
def load_artifacts():
    with open("model/artifacts/lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/artifacts/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open("model/artifacts/item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)
    with open("model/artifacts/user_vec.pkl", "rb") as f:
        user_vec = pickle.load(f)
    with open("model/artifacts/item_vec.pkl", "rb") as f:
        item_vec = pickle.load(f)
    return model, user_encoder, item_encoder, user_vec, item_vec

# === Load side features ===
def load_features():
    user_df = pd.read_csv("data_pipeline/processed/user_features.csv")
    item_df = pd.read_csv("data_pipeline/processed/item_features.csv")
    return user_df, item_df

# === Load item names ===
def load_item_names():
    try:
        df = pd.read_csv("data_pipeline/raw/item_names.csv")
        return df.set_index("itemid")["name"].to_dict()
    except FileNotFoundError:
        df = pd.read_csv("data_pipeline/raw/item_properties_part.csv")
        df = df[df["property"] == "name"]
        df = df.sort_values("timestamp").drop_duplicates("itemid", keep="last")
        return df.set_index("itemid")["value"].to_dict()

# === Recommend items ===
def recommend_items(visitorid, top_n=5):
    model, user_encoder, item_encoder, user_vec, item_vec = load_artifacts()
    user_df, item_df = load_features()
    item_names = load_item_names()

    try:
        visitorid_int = int(visitorid)
        user_idx = user_encoder.transform([visitorid_int])[0]
    except ValueError:
        print(f"Visitor ID {visitorid} not found in training data.")
        return

    # === Prepare user features ===
    user_row = user_df[user_df["visitorid"] == visitorid_int]
    if user_row.empty:
        print(f"No user features found for visitor {visitorid}")
        return

    user_feat_dict = user_row[["total_views", "total_purchases", "unique_items_viewed"]].to_dict("records")[0]
    user_features = user_vec.transform([user_feat_dict])

    # === Prepare item features ===
    all_item_ids = item_encoder.classes_
    item_rows = item_df[item_df["itemid"].isin(all_item_ids.astype(int))].set_index("itemid")
    item_rows = item_rows.loc[all_item_ids.astype(int)]
    item_feat_dicts = item_rows[["item_total_views", "item_total_purchases", "item_purchase_rate"]].to_dict("records")
    item_features = item_vec.transform(item_feat_dicts)

    # === Predict scores ===
    user_features_tiled = sp.vstack([user_features] * len(all_item_ids))
    scores = model.predict(
        user_ids=np.zeros(len(all_item_ids), dtype=int),
        item_ids=np.arange(len(all_item_ids)),
        user_features=user_features_tiled,
        item_features=item_features
    )

    # === Top-N items ===
    top_items = np.argsort(-scores)[:top_n]
    recommended_itemids = item_encoder.inverse_transform(top_items)

    print(f"\nTop {top_n} Recommended Items for Visitor {visitorid}:\n")
    for i, (itemid, score) in enumerate(zip(recommended_itemids, scores[top_items])):
        item_name = item_names.get(int(itemid), "Unknown Product")
        print(f"{i+1}. Item ID: {itemid} | Score: {score:.4f} | Name: {item_name}")

# === Run ===
if __name__ == "__main__":
    visitor_id = input("Enter visitorid to get recommendations: ").strip()
    recommend_items(visitor_id, top_n=5)
