📖 Data Prep Workflow for Sequence Models
1. Build raw EXIT sequences

This step creates sliding windows + labels for the exit model only.

python -m scripts.sequence.build_exit_sequence_dataset


Input: prices (raw OHLCV + indicators)

Output: MongoDB collection → exit_sequences

2. Build final training datasets (Entry + Exit)

This step creates the entry and exit training arrays from MongoDB and saves them to disk.

python -m scripts.build_training_dataset


Pulls entry data from training collection

Pulls exit data from exit_sequences collection

Builds sliding windows (with filters)

Saves compressed dataset:

data/sequence_dataset.npz
containing:

X_entry, y_entry, tickers_entry

X_exit, y_exit, tickers_exit

3. Train models

train_sequence_entry.py → trains entry model on X_entry, y_entry.

train_sequence_exit.py → trains exit model on X_exit, y_exit.

👉 So yes:

build_exit_sequence_dataset.py is mandatory to refresh the exit side.

build_training_dataset.py is the final collector/packager for both entry + exit arrays.

