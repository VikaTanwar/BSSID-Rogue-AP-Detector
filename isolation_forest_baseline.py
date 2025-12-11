import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# 1) Data load
df = pd.read_csv("wifi_networks_dataset.csv")

# Target (label) and features
y = df["is_rogue"]
X = df.drop(columns=["is_rogue", "label"])

print("Total:", len(df), "| Normal:", (y == 0).sum(), "| Rogue:", (y == 1).sum())

# 2) Train / Val split (70 / 30) with stratify
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Train size:", X_train.shape[0], "Val size:", X_val.shape[0])

# 3) Columns define karo
# Numeric columns (jisko scale karenge)
num_cols = ["signal_strength", "channel", "frequency"]
# Categorical columns (jisko one-hot encode karenge)
cat_cols = ["encryption", "vendor"]

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# 4) Numeric part
X_train_num = X_train[num_cols].copy()
X_val_num   = X_val[num_cols].copy()

# 5) Categorical part – missing ko 'Unknown' se fill
X_train_cat = X_train[cat_cols].fillna("Unknown")
X_val_cat   = X_val[cat_cols].fillna("Unknown")

# 6) One-hot encoding (train + val ko ek sath leke taaki same columns bane)
full_cat = pd.concat([X_train_cat, X_val_cat], axis=0)
full_cat_dummies = pd.get_dummies(full_cat, columns=cat_cols, drop_first=True)

# Encoded train/val categorical frames
X_train_cat_enc = full_cat_dummies.iloc[:len(X_train)].reset_index(drop=True)
X_val_cat_enc   = full_cat_dummies.iloc[len(X_train):].reset_index(drop=True)

# 7) Final feature matrix: numeric + encoded categorical
X_train_final = pd.concat(
    [X_train_num.reset_index(drop=True), X_train_cat_enc],
    axis=1
)
X_val_final = pd.concat(
    [X_val_num.reset_index(drop=True), X_val_cat_enc],
    axis=1
)

print("Final train shape:", X_train_final.shape)
print("Final val shape  :", X_val_final.shape)

# 8) Scale sirf numeric columns (baaki dummies as-is rahenge)
scaler = StandardScaler()

X_train_scaled = X_train_final.copy()
X_val_scaled   = X_val_final.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train_final[num_cols])
X_val_scaled[num_cols]   = scaler.transform(X_val_final[num_cols])

# 9) Isolation Forest – best config jo tumne tune kiya tha
iso = IsolationForest(
    contamination=0.15,   # from tuning
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_scaled)

# 10) Prediction on validation (-1 = anomaly/rogue, 1 = normal)
val_raw_pred = iso.predict(X_val_scaled)
y_val_pred = (val_raw_pred == -1).astype(int)

# 11) Evaluation
print("\nConfusion Matrix (val):")
print(confusion_matrix(y_val, y_val_pred))

print("\nClassification Report (val):")
print(classification_report(y_val, y_val_pred, digits=3))
