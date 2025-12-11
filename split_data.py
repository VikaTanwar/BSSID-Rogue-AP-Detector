import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Data load
df = pd.read_csv("wifi_networks_dataset.csv")

# Target
y = df["is_rogue"]
X = df.drop(columns=["is_rogue", "label"])

print("Total:", len(df), "| Normal:", (y==0).sum(), "| Rogue:", (y==1).sum())

# 2) Pehle train + temp (val+test) split (70% / 30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# 3) Phir temp ko val + test me split (15% / 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Train size:", X_train.shape[0])
print("Val size  :", X_val.shape[0])
print("Test size :", X_test.shape[0])

# 4) CSVs save karo taaki baar‑baar reuse ho sake
train = X_train.copy()
train["is_rogue"] = y_train

val = X_val.copy()
val["is_rogue"] = y_val

test = X_test.copy()
test["is_rogue"] = y_test

train.to_csv("wifi_train.csv", index=False)
val.to_csv("wifi_val.csv", index=False)
test.to_csv("wifi_test.csv", index=False)

print("Saved: wifi_train.csv, wifi_val.csv, wifi_test.csv")
