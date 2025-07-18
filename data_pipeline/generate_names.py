import pandas as pd
import os
import random
from itertools import product
from tqdm import tqdm

# Load item IDs
df = pd.read_csv("data_pipeline/processed/training_data.csv")
unique_itemids = df["itemid"].unique()

# Expanded name components to ensure at least 12,025 unique combinations
adjectives = [
    "Ultra", "Pro", "Smart", "Classic", "Elite", "Compact", "Advanced", "Max", "Eco", "Modern",
    "Prime", "Bold", "Next", "Super", "Lite", "Rapid", "Fusion", "Nova", "Quantum", "Dynamic",
    "Streamlined", "Turbo", "Versatile", "Minimal", "Reliable", "Sleek", "Agile", "Efficient", "Swift", "Solid"
]

brands = [
    "Zenith", "NovaTech", "UrbanEdge", "Aero", "VitaCore", "HyperX", "Omni", "Flexi", "NextGen", "CoreX",
    "Skyline", "Titan", "Vertex", "Glide", "Nimbus", "Solace", "Axon", "Delta", "Luma", "Echo",
    "Neon", "Orbit", "Matrix", "QuantumEdge", "Strato", "Infini", "Helio", "Zentra", "Kinetic", "Motiv"
]

products = [
    "Smartphone", "Sneakers", "Backpack", "Headphones", "Watch", "Laptop", "Blender", "Chair", "Tablet", "Jacket",
    "Microwave", "Camera", "Desk", "Speaker", "Fridge", "Monitor", "Keyboard", "Treadmill", "Router", "Toaster",
    "Mouse", "Cooker", "Fan", "Light", "Bottle", "Sofa", "Bicycle", "Heater", "Printer", "Projector"
]

# Generate all possible unique name combinations
all_combinations = list(product(adjectives, brands, products))
random.shuffle(all_combinations)

# Check if we have enough combinations
if len(all_combinations) < len(unique_itemids):
    raise ValueError(
        f"Not enough unique name combinations ({len(all_combinations)}) "
        f"for {len(unique_itemids)} itemids. Expand your name components."
    )

# Create mapping from itemid to unique name
item_name_map = {}
for itemid, (adj, brand, product_name) in tqdm(
    zip(unique_itemids, all_combinations),
    total=len(unique_itemids),
    desc="Generating item names"
):
    item_name_map[itemid] = f"{adj} {brand} {product_name}"

# Save to CSV
os.makedirs("data_pipeline/processed", exist_ok=True)
mapping_df = pd.DataFrame({
    "itemid": list(item_name_map.keys()),
    "name": list(item_name_map.values())
})
mapping_df.to_csv("data_pipeline/processed/item_names.csv", index=False)

print("Generated realistic product names in data_pipeline/processed/item_names.csv")
