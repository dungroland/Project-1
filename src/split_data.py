import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu đã làm sạch
df = pd.read_csv("../data/processed/cleaned_dataset.csv")

# 2. Chia dữ liệu: 80% để học (train), 20% để thi (test)
train_df, test_df = train_test_split(
    df,
    test_size=0.2, 
    random_state=42, 
    stratify=df['sentiment']
)

# 3. Lưu thành 2 file riêng biệt
train_df.to_csv("../data/processed/train.csv", index=False)
test_df.to_csv("../data/processed/test.csv", index=False)

print(f"Hoàn thành chia dữ liệu!")
print(f"Số lượng mẫu huấn luyện (Train): {len(train_df)}")
print(f"Số lượng mẫu kiểm tra (Test): {len(test_df)}")