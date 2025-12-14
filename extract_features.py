import os
import csv
import pandas as pd

DATASET_DIR = "dataset"
FEATURES_DIR = "features"
OUTPUT_CSV = os.path.join(FEATURES_DIR, "features.csv")

os.makedirs(FEATURES_DIR, exist_ok=True)

rows = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        with open(os.path.join(DATASET_DIR, file)) as f:
            reader = csv.reader(f)
            for r in reader:
                if len(r) == 69:
                    rows.append(list(map(float, r)) + [label])

df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(69)] + ["label"])
df.to_csv(OUTPUT_CSV, index=False)
