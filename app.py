import streamlit as st
import joblib, os, json
import pandas as pd
import matplotlib.pyplot as plt
import uuid

# Load model & vectorizer
model_path = "models/best_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Please train the model first by running `python run.py`.")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.set_page_config(page_title="Fake News Detector -by Vishal", page_icon="🧠")
st.title("🧠 Fake News Detector -by Vishal")
st.caption("Offline model trained on news articles (Real vs Fake)")

# User input
text_input = st.text_area("📄 Paste a news article:", height=200)

# Initialize predicted label
label = None

if st.button("🔍 Predict"):
    if text_input.strip():
        vec = vectorizer.transform([text_input])
        pred = model.predict(vec)[0]
        label = "🟢 REAL News" if pred == 1 else "🔴 FAKE News"
        st.subheader("Prediction:")
        st.success(label)
    else:
        st.warning("Please enter some text.")

# Optional: Add expected answer input & save for evaluation
if text_input.strip() and label:
    st.markdown("---")
    st.subheader("📝 Evaluation Logging (Optional)")
    expected = st.text_input("Expected Label (REAL or FAKE)").strip().upper()

    if expected in {"REAL", "FAKE"}:
        if st.button("💾 Save for Evaluation"):
            os.makedirs("eval_logs", exist_ok=True)
            log_data = {
                "input_text": text_input,
                "predicted_label": "REAL" if "REAL" in label else "FAKE",
                "expected_label": expected
            }
            filename = f"eval_logs/eval_{uuid.uuid4().hex}.json"
            with open(filename, "w") as f:
                json.dump(log_data, f, indent=2)
            st.success(f"✅ Evaluation saved to `{filename}`")
    else:
        st.info("Enter 'REAL' or 'FAKE' to enable save.")

st.markdown("---")
st.header("📊 Model Evaluation Metrics")

# Load and show logs
log_dir = "logs"
log_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]

if not log_files:
    st.info("No logs found. Run `python run.py` to generate logs.")
else:
    logs = []
    for file in sorted(log_files):
        with open(os.path.join(log_dir, file)) as f:
            data = json.load(f)
            data["run"] = file.replace(".json", "")
            logs.append(data)

    df = pd.DataFrame(logs).sort_values("run")

    selected_metrics = st.multiselect("Select metrics to display:", ["accuracy", "precision", "recall", "f1_score"], default=["accuracy", "f1_score"])

    if selected_metrics:
        st.line_chart(df.set_index("run")[selected_metrics])
