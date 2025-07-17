from feast import FeatureView, Field
from feast.types import Int64, Float32
from feast import FileSource
from datetime import timedelta
from .entity import user, item

user_source = FileSource(
    path="data_pipeline/processed/user_features.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

user_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="total_views", dtype=Int64),
        Field(name="total_add_to_cart", dtype=Int64),
        Field(name="total_purchases", dtype=Int64),
        Field(name="unique_items_viewed", dtype=Int64),
    ],
    source=user_source,
)

# ITEM FEATURES

item_source = FileSource(
    path="data_pipeline/processed/item_features.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column=None,
)

item_fv = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=30),
    schema=[
        Field(name="item_total_views", dtype=Int64),
        Field(name="item_total_purchases", dtype=Int64),
        Field(name="item_purchase_rate", dtype=Float32),
    ],
    source=item_source,
)
