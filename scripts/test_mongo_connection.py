# scripts/test_mongo_connection.py
from core.db import db, ping

print("Pinging Mongo...")
print("Ping ok?", ping())
print("DB name:", db.name)
print("Collections:", db.list_collection_names())
