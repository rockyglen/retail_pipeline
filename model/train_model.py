import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from lightfm import LightFM
from sklearn.preprocessing import LabelEncoder
import os
import json

data_path = "data_pipeline/processed/training_data.csv"
df = pd.read_csv(data_path)

# Step 1: Encode user and item IDs to numerical values
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["user_idx"] = user_encoder.fit_transform(df["visitorid"])
df["item_idx"] = item_encoder.fit_transform(df["itemid"])

# Step 2: Create interaction matrix (user x item) with implicit feedback

df["interaction"] = 1
num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()

interaction_matrix = sp.coo_matrix(
    (df["interaction"], (df["user_idx"], df["item_idx"])), shape=(num_users, num_items)
)

# Step 3: Initialize and train the model

model=LightFM(loss='warp')

model.fit(interaction_matrix,epochs=10,num_threads=4)

# Step 4: Save the model and encoders
os.makedirs("model/artifacts", exist_ok=True)

# Step 5: Create and save ID maps
user_id_map = dict(zip(df["visitorid"], df["user_idx"]))
item_id_map = dict(zip(df["itemid"], df["item_idx"]))


with open("model/artifacts/user_id_map.json", "w") as f:
    json.dump(user_id_map, f)

with open("model/artifacts/item_id_map.json", "w") as f:
    json.dump(item_id_map, f)

with open("model/artifacts/lightfm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/artifacts/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)

with open("model/artifacts/item_encoder.pkl", "wb") as f:
    pickle.dump(item_encoder, f)

print("Model training complete. Artifacts saved to model/artifacts/")
