import numpy as np
import pandas as pd
import pickle
import json
from lightfm import LightFM
from scipy.sparse import coo_matrix


def load_artifacts():
    with open("model/artifacts/lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/artifacts/user_id_map.json", "r") as f:
        user_id_map = json.load(f)

    with open("model/artifacts/item_id_map.json", "r") as f:
        item_id_map = json.load(f)

    return model, user_id_map, item_id_map


def build_sparse_matrix(data, user_id_map, item_id_map):
    row = data["visitorid"].map(user_id_map)
    col = data["itemid"].map(item_id_map)
    interactions = coo_matrix(
        (np.ones(len(data)), (row, col)), shape=(len(user_id_map), len(item_id_map))
    )
    return interactions


def recommend_items(visitorid,top_n=5):
    model,user_id_map,item_id_map=load_artifacts()

    if str(visitorid) not in user_id_map:
        raise ValueError(f"Visitor ID {visitorid} not found in training data.")
    
    df=pd.read_csv("model/joined_features.csv")

    # Build the interaction matrix
    interaction_matrix = build_sparse_matrix(df, user_id_map, item_id_map)

    # Get internal user index
    user_index = user_id_map[str(visitorid)]

    # Predict scores for all items
    scores = model.predict(user_index, np.arange(len(item_id_map)))

    # Rank items by score
    top_indices = np.argsort(-scores)[:top_n]

    # Map internal item IDs back to original itemids
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    recommended_item_ids = [reverse_item_map[i] for i in top_indices]

    return recommended_item_ids

if __name__ == "__main__":
    visitor_id = input("Enter visitorid to get recommendations: ")
    recommendations = recommend_items(visitor_id, top_n=5)
    print("Top 5 Recommended itemids:", recommendations)
