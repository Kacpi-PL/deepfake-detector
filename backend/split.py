import pandas as pd
import os
from sklearn.model_selection import train_test_split

CSV = "metadata.csv"
PATH = "../data/real-vs-fake-faces-stylegan3"
df = pd.read_csv(os.path.join(PATH, CSV))

# Dict for mapping and mapping of label column
LABEL_MAP = {"fake": 0, "real": 1}
df["label"] = df["label"].map(LABEL_MAP)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv(os.path.join(PATH, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(PATH, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(PATH, "test_split.csv"), index=False)
train_df = pd.read_csv(os.path.join(PATH, "train_split.csv"))
val_df = pd.read_csv(os.path.join(PATH, "val_split.csv"))
test_df = pd.read_csv(os.path.join(PATH, "test_split.csv"))