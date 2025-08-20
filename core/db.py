"""
core/db.py

Central MongoDB connection module.
Loads URI from .env and connects to the specified database.
Exposes `db` and key collections like `prices` and `fundamentals`.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env file from project root
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "trading_app")

if not MONGODB_URI:
    raise RuntimeError("❌ MONGODB_URI is not set in .env")

# Connect to MongoDB (local or remote)
_client = MongoClient(MONGODB_URI, uuidRepresentation="standard")
db = _client[DB_NAME]

# Common collections
prices = db["prices"]
fundamentals = db["fundamentals"]
training = db["training"]
latest = db["latest"]

# Optional dev log
print(f"✅ Connected to MongoDB: {DB_NAME}")
