import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from pydantic import BaseModel
import csv
from pathlib import Path
from datetime import datetime
import json, os

TRUSTED_FILE = Path("trusted_networks.csv")

app = FastAPI(title="BSSID Rogue AP Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "https://vikatanwar.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

LAST_SCAN_FILE = "last_scan.json"

@app.get("/last-scan")
def get_last_scan():
    if not os.path.exists(LAST_SCAN_FILE):
        return {"detail": "no scan yet"}
    with open(LAST_SCAN_FILE, "r") as f:
        data = json.load(f)
    return data

df = pd.read_csv("wifi_networks_dataset.csv")

X = df[["signal_strength", "channel", "frequency", "encryption", "vendor"]]
y = df["is_rogue"]

numeric_cols = ["signal_strength", "channel", "frequency"]
cat_cols = ["encryption", "vendor"]

preprocessor = ColumnTransformer(
    [
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=10,
            random_state=42
        )),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

class WifiScan(BaseModel):
    bssid: str
    ssid: str
    signal_strength: float
    channel: int
    frequency: int
    encryption: str
    vendor: str

class TrustRequest(BaseModel):
    bssid: str
    ssid: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_accuracy": round(acc, 3)
    }

@app.post("/mark_trusted")
def mark_trusted(req: TrustRequest):
    exists = TRUSTED_FILE.exists()
    with TRUSTED_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["bssid", "ssid", "first_seen", "notes"])
        writer.writerow([req.bssid, req.ssid, datetime.now().isoformat(), "user-confirmed"])
    return {"status": "ok"}

@app.post("/predict")
def predict(scan: WifiScan):
    row = pd.DataFrame(
        [
            {
                "signal_strength": scan.signal_strength,
                "channel": scan.channel,
                "frequency": scan.frequency,
                "encryption": scan.encryption,
                "vendor": scan.vendor,
            }
        ]
    )
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]
    label = "Rogue" if pred == 1 else "Normal"
    return {
        "bssid": scan.bssid,
        "ssid": scan.ssid,
        "prediction": label,
        "probability_rogue": round(float(proba), 3)
    }
