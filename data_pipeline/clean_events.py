import pandas as pd

import os

raw_path = "data_pipeline/raw/events.csv"

events = pd.read_csv(raw_path)
# Step 1: Filter event types (we only want view, addtocart, transaction)
valid_events = ["view", "addtocart", "transaction"]
events = events[events["event"].isin(valid_events)]

# Step 2: Convert timestamp (Unix → datetime)
events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")

# Step 3: Drop rows with missing user_id or item_id
events.dropna(subset=['visitorid','itemid'],inplace=True)

# Step 4: Sort by time
events=events.sort_values('timestamp')

# Step 5: Save to processed folder

os.makedirs("data_pipeline/processed",exist_ok=True)

events.to_csv("data_pipeline/processed/cleaned_events.csv",index=False)

print(f"Cleaned dataset saved with {len(events)} rows.")

