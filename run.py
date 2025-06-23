# run.py
import pandas as pd
import os, joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from core.trainer import train_model
from core.evaluator import evaluate_model
from core.utils import log_metrics

# 1. Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
fake["label"], true["label"] = 0, 1

# 2. Combine and shuffle
df = pd.concat([fake, true]).sample(frac=1, random_state=42)
X, y = df["text"], df["label"]

# 3. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=200)
trained_model = train_model(model, X_train_vec, y_train)

# 5. Evaluate
results = evaluate_model(trained_model, X_test_vec, y_test)
print("📊 Evaluation Metrics:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# 6. Save model & vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(trained_model, "models/best_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# 7. Log metrics
log_metrics(results)