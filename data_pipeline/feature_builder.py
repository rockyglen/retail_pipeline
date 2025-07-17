import pandas as pd
import os
from datetime import datetime

events = pd.read_csv("data_pipeline/processed/cleaned_events.csv")

events["timestamp"] = pd.to_datetime(events["timestamp"])

user_features = (
    events.groupby("visitorid").agg(
        total_views=pd.NamedAgg(column="event", aggfunc=lambda x: (x == "view").sum()),
        total_add_to_cart=pd.NamedAgg(
            column="event", aggfunc=lambda x: (x == "addtocart").sum()
        ),
        total_purchases=pd.NamedAgg(
            column="event", aggfunc=lambda x: (x == "transaction").sum()
        ),
        unique_items_viewed=pd.NamedAgg(column="itemid", aggfunc=lambda x: x.nunique()),
    )
).reset_index()

item_grouped = events.groupby("itemid")
item_features = pd.DataFrame(
    {
        "itemid": item_grouped.size().index,
        "item_total_views": item_grouped["event"].apply(lambda x: (x == "view").sum()),
        "item_total_purchases": item_grouped["event"].apply(
            lambda x: (x == "transaction").sum()
        ),
    }
).reset_index(drop=True)

item_features["item_purchase_rate"] = item_features["item_total_purchases"] / (
    item_features["item_total_views"] + 1
)

now = datetime.utcnow()
user_features['event_timestamp']=now
item_features['event_timestamp']=now

os.makedirs("data_pipeline/processed", exist_ok=True)
user_features.to_csv("data_pipeline/processed/user_features.csv", index=False)
item_features.to_csv("data_pipeline/processed/item_features.csv", index=False)

print("Features saved successfully.")
