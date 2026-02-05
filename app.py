import streamlit as st
import joblib
import re

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Preprocess function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# UI
st.set_page_config(page_title="Sentiment Classification App", layout="centered")

st.title("Sentiment Classification App")
st.write("Phân loại cảm xúc (Positive / Negative) using TF-IDF & Logistic Regression")

user_input = st.text_area("Nhập đánh giá phim:", height=150)

if st.button("🔍 Dự đoán cảm xúc"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung đánh giá!")
    else:
        clean_text = preprocess_text(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        if prediction == 1:
            st.success(f"✅ Positive (Xác suất: {probability[1]*100:.2f}%)")
        else:
            st.error(f"❌ Negative (Xác suất: {probability[0]*100:.2f}%)")

st.markdown("---")
st.caption("Demo app for Sentiment Classification project")
