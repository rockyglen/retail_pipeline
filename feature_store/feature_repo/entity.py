from feast import Entity
from feast.value_type import ValueType

user = Entity(
    name="visitorid",
    value_type=ValueType.INT64,
    description="User ID"
)

item=Entity(
    name="itemid",
    value_type=ValueType.INT64,
    description="Item ID"
)