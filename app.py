import streamlit as st
import joblib
import re

# ===============================
# 1. Load models & vectorizer
# ===============================
logreg_model = joblib.load("models/logreg_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ===============================
# 2. Text cleaning function
# ===============================
def fast_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters & spaces
    return re.sub(r"\s+", " ", text).strip()

# ===============================
# 3. Streamlit UI
# ===============================
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬")
st.title("💬 Sentiment Analysis System")
st.write("This app predicts whether a review is **Positive** or **Negative** using Logistic Regression & SVM models.")

# Input box
user_input = st.text_area("✍️ Enter your review here:")

if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Clean input
        cleaned_text = fast_clean(user_input)

        # Transform using TF-IDF
        text_vec = vectorizer.transform([cleaned_text])  # ✅ FIX: 2D array

        # Predictions
        logreg_pred = logreg_model.predict(text_vec)[0]
        svm_pred = svm_model.predict(text_vec)[0]

        # Display results
        st.subheader("📌 Prediction Results")
        st.write(f"**Logistic Regression Prediction:** {logreg_pred}")
        st.write(f"**SVM Prediction:** {svm_pred}")

        # Extra style
        if logreg_pred == "positive" or svm_pred == "positive":
            st.success("😊 This review seems **Positive**!")
        else:
            st.error("☹️ This review seems **Negative**.")
