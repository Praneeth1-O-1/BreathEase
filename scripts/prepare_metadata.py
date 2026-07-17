import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "data/raw/coughvid_v3/public_dataset_v3/coughvid_20211012"

CSV_PATH = (
    "data/raw/coughvid_v3/tabular_form/"
    "tabular_form/coughvid_v3.csv"
)

OUTPUT_DIR = "metadata"
COUGH_THRESHOLD = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load metadata
# -----------------------------
df = pd.read_csv(CSV_PATH)

print(f"Original samples : {len(df)}")

df = df[
    ~(
        df["status"].isna()
        &
        df["respiratory_condition"].isna()
    )
]

print(f"After removing unknown labels : {len(df)}")


def create_label(row):

    if (
        row["status"] == "healthy"
        and row["respiratory_condition"] == False
    ):
        return 0

    if (
        row["status"] in ["symptomatic", "COVID-19"]
        or row["respiratory_condition"] == True
    ):
        return 1

    return None


df["label"] = df.apply(create_label, axis=1)

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print(df["label"].value_counts())

# -----------------------------
# Keep only confident coughs
# -----------------------------
df = df[df["cough_detected"] >= COUGH_THRESHOLD]

print(f"After cough confidence filter : {len(df)}")

print(df["label"].value_counts())


# -----------------------------
# Train / Validation / Test Split
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

print("\nDataset Split")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# -----------------------------
# Save CSVs
# -----------------------------
train_df.to_csv("metadata/train.csv", index=False)
val_df.to_csv("metadata/val.csv", index=False)
test_df.to_csv("metadata/test.csv", index=False)

print("\nCSV files saved successfully!")