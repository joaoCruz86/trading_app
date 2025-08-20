# scripts/utils/clear_collections.py

from core.db import db

# List of MongoDB collections to clear
collections_to_clear = [
    "prices",
    "fundamentals",
    "training",
    "latest",
    "signals",
    "historical"
]

for collection_name in collections_to_clear:
    result = db[collection_name].delete_many({})
    print(f"🧹 Cleared '{collection_name}': {result.deleted_count} documents removed.")
