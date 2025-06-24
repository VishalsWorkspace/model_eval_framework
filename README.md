
# 🚀 Model Evaluation Framework for Fake News Detection

**Automated ML pipeline using TF‑IDF & Logistic Regression with Streamlit UI & Logging**

---

## 📌 What Is This Project?

The **Model Evaluation Framework for Fake News Detection** is an end‑to‑end offline system that:

- Loads a **real-world Fake vs Real news dataset**
- Preprocesses text using **TF‑IDF vectorization**
- Trains a **Logistic Regression** classifier
- Evaluates performance with **accuracy, precision, recall, F1-score**
- Logs all metrics as timestamped JSON files in `logs/`
- Saves both the **trained model** and **vectorizer** as `.pkl` files for reuse
- Includes an interactive **Streamlit** UI (`app.py`) for classification and metrics visualization

This framework is designed for easy adaptation to other ML tasks or LLM output benchmarking, enabling you to track model performance over time and compare versions.

---

## 🌟 Why It Matters

- **Real-world payoff**: Fake news detection is a critical and impactful application in AI-driven media analysis, content moderation, and misinformation prevention.
- **Modular & Transparent**: Clear structure separates training, evaluation, logging, and UI—ideal for collaborative development and scalability.
- **Fully Offline & Lightweight**: No cloud/API needed—works with standard CPU setups.
- **Benchmark-ready**: Save logs and models to track and compare growth over time or evaluate new candidate models.

---

## 🛠️ Tech Stack

| Component               | Description                              |
|------------------------|------------------------------------------|
| 🐍 Python 3.9+         | Core development language                |
| 🧮 scikit‑learn         | Model, metrics, splitting, TF‑IDF        |
| 📦 Pandas, NumPy        | Data handling and transformations        |
| 💾 joblib               | Model & vectorizer serialization        |
| 📈 Matplotlib           | Metrics visualization                   |
| ⚙️ Streamlit            | Interactive web interface               |
| 📊 Optional: psutil     | System monitoring (CPU/RAM logging)     |

---

## 🏁 Get Started

### 1. Clone the repo & enter directory
```bash
git clone https://github.com/VishalsWorkspace/model_eval_framework.git
cd model_eval_framework

2. Install dependencies
python -m venv venv
source venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

3. Add the dataset (if not already present)
Place two files in data/:

Fake.csv (fake news samples)

True.csv (real news samples)

4. Train the model + evaluate
python run.py

This will:
Preprocess and vectorize data
Train logistic regression
Output scores in terminal

Save:
models/best_model.pkl
models/tfidf_vectorizer.pkl
JSON metrics in logs/

5. Launch the UI
streamlit run app.py

You’ll get:
A web page to paste a news article and get a Real or Fake prediction
An interactive line chart showing historical metrics from past runs

🔍 Project Structure
graphql
Copy
Edit
model_eval_framework/
├── core/
│   ├── trainer.py        # Training logic
│   ├── evaluator.py      # Metric evaluation
│   ├── tuner.py          # (Optional) Hyperparameter tuning
│   └── utils.py          # Logging functions
├── data/                 # `Fake.csv` and `True.csv`
├── logs/                 # JSON metric logs (time-stamped)
├── models/               # Saved `.pkl` model & vectorizer
├── run.py                # Full pipeline runner
├── app.py                # Streamlit UI app
├── requirements.txt      # Dependencies
└── README.md             # This documentation


🚀 Extend or Contribute:
Add advanced models (e.g., SVM, random forest, neural nets)

Tune hyperparameters with tuner.py and tools like Optuna

Enable file uploads in Streamlit for custom datasets

Add system monitoring with psutil in UI to show CPU/RAM usage

Provide unit tests and CI/CD setup

Add Docker support for easy deployment

👏 Contribution
This project is open-source and welcomes contributions!


📞 Stay in Touch
Maintained by Vishal
Feel free to open issues or submit PRs for features, improvements, or bug fixes.
Happy collaborating!