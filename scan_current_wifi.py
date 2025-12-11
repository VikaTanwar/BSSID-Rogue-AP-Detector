import subprocess
import re
import requests
import json
import os

API_URL =  "https://bssid-rogue-ap-detector.onrender.com/predict"
LAST_SCAN_FILE = "last_scan.json"

print("Reading current WiFi interface info...")

cmd = ["netsh", "wlan", "show", "interfaces"]
result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
output = result.stdout

if "State" not in output or "connected" not in output.lower():
    print("Not connected to any WiFi network.")
    exit()

def find(pattern, default=""):
    m = re.search(pattern, output, re.IGNORECASE)
    return m.group(1).strip() if m else default

ssid = find(r"SSID\s*:\s*(.+)")
bssid = find(r"BSSID\s*:\s*(.+)")
signal_str = find(r"Signal\s*:\s*(\d+)")
auth = find(r"Authentication\s*:\s*(.+)")

if not ssid or not bssid:
    print("Could not parse SSID/BSSID from netsh output.")
    print(output)
    exit()

signal_percent = int(signal_str) if signal_str else 50
signal_dbm = -30 + (signal_percent - 100) * 0.6

encryption = auth
vendor = "Generic"
channel = 6
frequency = 2437

payload = {
    "bssid": bssid,
    "ssid": ssid,
    "signal_strength": signal_dbm,
    "channel": channel,
    "frequency": frequency,
    "encryption": encryption,
    "vendor": vendor,
}

print("Payload to API:")
print(payload)

try:
    resp = requests.post(API_URL, json=payload, timeout=5)
    data = resp.json()
    print("\nModel decision:")
    print("Prediction:", data.get("prediction"))
    print("Rogue probability:", data.get("probability_rogue"))

    payload["prediction"] = data.get("prediction")
    payload["probability_rogue"] = data.get("probability_rogue")

    with open(LAST_SCAN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"\nLast scan saved to {os.path.abspath(LAST_SCAN_FILE)}")

except Exception as e:
    print("Error calling API:", e)
