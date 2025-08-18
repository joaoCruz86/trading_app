import pandas as pd
from core.db import db

# Load a few rows from the MongoDB training collection
rows = list(db["training"].find().limit(10))
df = pd.DataFrame(rows)

# Optional: Drop the MongoDB _id field for display
df.drop(columns=["_id"], errors="ignore", inplace=True)

# Print the dataframe
print("\n📋 Sample training data:")
print(df.head(10).to_string(index=False))
