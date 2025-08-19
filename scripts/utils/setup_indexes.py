from core.db import db, ensure_indexes, ping

print("Pinging Mongo...")
print("Ping ok?", ping())

print("Creating indexes...")
ensure_indexes()

print("prices indexes:", list(db.prices.index_information().keys()))
print("fundamentals indexes:", list(db.fundamentals.index_information().keys()))
print("Done ✅")
