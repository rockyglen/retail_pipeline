import pandas as pd
from feast import FeatureStore
import datetime

interactions = pd.read_csv("data_pipeline/processed/interactions.csv")

store = FeatureStore(repo_path="feature_store")

user_df = interactions[["visitorid"]].drop_duplicates()
item_df = interactions[["itemid"]].drop_duplicates()

user_df["event_timestamp"] = datetime.datetime.utcnow()
item_df["event_timestamp"] = datetime.datetime.utcnow()

user_features = store.get_historical_features(
    entity_df=user_df,
    features=["user_features:total_views", "user_features:total_purchases"],
).to_df()

item_features = store.get_historical_features(
    entity_df=item_df,
    features=[
        "item_features:item_total_views",
        "item_features:item_total_purchases",
        "item_features:item_purchase_rate",
    ],
).to_df()

df = interactions.merge(user_features, on="visitorid", how="left")
df = df.merge(item_features, on="itemid", how="left")

df.to_csv("data_pipeline/processed/training_data.csv", index=False)

df.to_csv("model/joined_features.csv", index=False)


print("Joined interaction+user/item features:", df.shape)
