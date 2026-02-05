import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Đọc dữ liệu đã chia
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 2. Biến đổi văn bản thành con số (Vectorization)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
    )
X_train = tfidf.fit_transform(train_df['review'].values.astype('U'))
X_test = tfidf.transform(test_df['review'].values.astype('U'))

y_train = train_df['sentiment']
y_test = test_df['sentiment']

# 3. Huấn luyện mô hình Logistic Regression 
print("Đang huấn luyện mô hình...")
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Kiểm tra độ chính xác
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {acc * 100:.2f}%")

# 5. Lưu mô hình và bộ biến đổi vào thư mục 'models'
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("Đã lưu mô hình vào thư mục models/")