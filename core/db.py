# core/db.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env from project root
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "trading_app")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set in .env")

# Works for local Mongo and Atlas URIs
_client = MongoClient(MONGODB_URI, uuidRepresentation="standard")
db = _client[DB_NAME]

def ping() -> bool:
    """Return True if the DB responds."""
    _client.admin.command("ping")
    return True

# Optional: run this file directly to test
if __name__ == "__main__":
    print("Ping ok?", ping())
    print("DB name:", db.name)

from pymongo import ASCENDING

def ensure_indexes():
    # One document per (ticker, date) in prices
    db.prices.create_index(
        [("ticker", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uniq_ticker_date"
    )
    db.prices.create_index([("date", ASCENDING)], name="date_idx")

    # One document per (ticker, period) in fundamentals
    db.fundamentals.create_index(
        [("ticker", ASCENDING), ("period", ASCENDING)],
        unique=True,
        name="uniq_ticker_period"
    )