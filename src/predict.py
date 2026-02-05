import joblib
import re

# 1. Tải mô hình và bộ biến đổi đã lưu
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_sentiment(text):
    # Làm sạch câu nhập vào
    cleaned = clean_text(text)
    # Biến đổi thành số (vector)
    vectorized = tfidf.transform([cleaned])
    # Dự đoán
    prediction = model.predict(vectorized)
    return "Tích cực (Positive)" if prediction[0] == 1 else "Tiêu cực (Negative)"

# 2. Chạy thử nghiệm
if __name__ == "__main__":
    print("--- Phân loại cảm xúc ---")
    while True:
        user_input = input("\nNhập 1 câu nhận xét hoặc 'exit' để thoát: ")
        if user_input.lower() == 'exit':
            break
        
        result = predict_sentiment(user_input)
        print(f"{result}")
        