import joblib

# Load
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sample inputs
news = [
    "Government releases official budget for the next fiscal year.",
    "NASA confirms alien spaceship landed on Mars yesterday."
]

vec = vectorizer.transform(news)
pred = model.predict(vec)

for text, label in zip(news, pred):
    print(f"\n📰 {text}\n➡️ Prediction: {'REAL' if label == 1 else 'FAKE'}")
