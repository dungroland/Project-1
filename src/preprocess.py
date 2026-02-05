import pandas as pd
import re
import os

# 1. Đọc dữ liệu
df = pd.read_csv("data/raw/dataset.csv")

# 2. Loại bỏ dòng thiếu dữ liệu
df = df.dropna(subset=['review', 'sentiment'])

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

print("Đang làm sạch văn bản...")
df['review'] = df['review'].apply(clean_text)

# 3. Map nhãn
label_map = {
    'positive': 1,
    'negative': 0
}
df['sentiment'] = df['sentiment'].map(label_map)

# 4. Lưu
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_dataset.csv", index=False)

print("Hoàn thành tiền xử lý!")
print(df.head())
