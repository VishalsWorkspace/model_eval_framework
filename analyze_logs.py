import os, json
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "logs/"
logs = []

for file in os.listdir(log_dir):
    if file.endswith(".json"):
        with open(os.path.join(log_dir, file)) as f:
            data = json.load(f)
            data["run"] = file.replace(".json", "")
            logs.append(data)

df = pd.DataFrame(logs).sort_values("run")
plt.figure(figsize=(10, 5))

for metric in ["accuracy", "precision", "recall", "f1_score"]:
    plt.plot(df["run"], df[metric], marker="o", label=metric)

plt.xticks(rotation=45)
plt.title("Model Evaluation Metrics Over Time")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
