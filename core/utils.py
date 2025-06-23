# utils.py
import os
import json
from datetime import datetime

def log_metrics(metrics: dict):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/run_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📁 Metrics saved to {path}")
