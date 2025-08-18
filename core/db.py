import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING   # import MongoClient first


# Load .env from project root
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "trading_app")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set in .env")

# Works for local Mongo and Atlas URIs
_client = MongoClient(MONGODB_URI, uuidRepresentation="standard")
db = _client[DB_NAME]

# Collections
prices = db["prices"]
fundamentals = db["fundamentals"]
