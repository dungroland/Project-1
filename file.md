# Giải Thích Toàn Bộ Dự Án Phân Loại Cảm Xúc (Sentiment Classification)

## 1. Tổng Quan Dự Án

Đây là một dự án Machine Learning hoàn chỉnh để phân loại cảm xúc của các đánh giá phim thành hai loại:
- **Positive (Tích cực)**: Đánh giá tích cực về phim
- **Negative (Tiêu cực)**: Đánh giá tiêu cực về phim

### Mục tiêu chính:
- Xây dựng pipeline ML hoàn chỉnh cho bài toán NLP
- Sử dụng TF-IDF để trích xuất đặc trưng từ văn bản
- Huấn luyện mô hình Logistic Regression
- Tạo ứng dụng web để dự đoán cảm xúc

## 2. Cấu Trúc Thư Mục

```
sentiment-classification/
├── data/
│   ├── raw/                    # Dữ liệu gốc từ Kaggle
│   │   └── dataset.csv
│   └── processed/              # Dữ liệu đã xử lý
│       ├── cleaned_dataset.csv
│       ├── train.csv
│       └── test.csv
├── models/                     # Mô hình đã huấn luyện
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/                        # Mã nguồn chính
│   ├── preprocess.py
│   ├── split_data.py
│   ├── train_model.py
│   └── predict.py
├── notebooks/                  # Jupyter notebooks thử nghiệm
│   ├── experiments.ipynb
│   └── visualization.ipynb
├── app.py                      # Ứng dụng Streamlit
├── requirements.txt            # Danh sách thư viện
└── README.md                   # Tài liệu dự án
```

## 3. Chi Tiết Từng File Code

### 3.1. File `src/preprocess.py` - Tiền Xử Lý Dữ Liệu

**Chức năng**: Làm sạch và chuẩn hóa dữ liệu văn bản thô

**Các bước thực hiện**:
1. **Đọc dữ liệu gốc**: Từ file `data/raw/dataset.csv`
2. **Loại bỏ dữ liệu thiếu**: Xóa các dòng không có nội dung review hoặc sentiment
3. **Làm sạch văn bản** (hàm `clean_text`):
   - Loại bỏ thẻ HTML: `<br />`, `<p>`, v.v.
   - Chuyển về chữ thường: `text.lower()`
   - Loại bỏ ký tự đặc biệt: chỉ giữ lại chữ cái và khoảng trắng
4. **Chuyển đổi nhãn**:
   - `'positive'` → `1`
   - `'negative'` → `0`
5. **Lưu kết quả**: Vào file `data/processed/cleaned_dataset.csv`

**Ví dụ**:
```
Input:  "This movie is <br />AMAZING!!!"
Output: "this movie is amazing"
```

### 3.2. File `src/split_data.py` - Chia Dữ Liệu

**Chức năng**: Chia dữ liệu thành tập huấn luyện và kiểm tra

**Tham số**:
- **Tỷ lệ chia**: 80% train, 20% test
- **random_state**: 42 (để kết quả có thể tái tạo)
- **stratify**: Đảm bảo tỷ lệ positive/negative giống nhau ở cả 2 tập

**Kết quả**:
- `data/processed/train.csv`: 40,000 mẫu
- `data/processed/test.csv`: 10,000 mẫu

### 3.3. File `src/train_model.py` - Huấn Luyện Mô Hình

**Chức năng**: Huấn luyện mô hình phân loại cảm xúc

**Các bước**:

1. **Vector hóa văn bản với TF-IDF**:
   ```python
   TfidfVectorizer(
       max_features=5000,    # Chỉ lấy 5000 từ quan trọng nhất
       ngram_range=(1, 2)    # Sử dụng cả unigram và bigram
   )
   ```

2. **Huấn luyện Logistic Regression**:
   ```python
   LogisticRegression(
       max_iter=1000,        # Số vòng lặp tối đa
       solver='lbfgs',       # Thuật toán tối ưu
       n_jobs=-1            # Sử dụng tất cả CPU cores
   )
   ```

3. **Đánh giá mô hình**: Tính độ chính xác trên tập test

4. **Lưu mô hình**: 
   - `models/sentiment_model.pkl`: Mô hình Logistic Regression
   - `models/tfidf_vectorizer.pkl`: Bộ vector hóa TF-IDF

**Kết quả**: Độ chính xác khoảng 89-90%

### 3.4. File `src/predict.py` - Dự Đoán Cảm Xúc

**Chức năng**: Sử dụng mô hình đã huấn luyện để dự đoán cảm xúc

**Quy trình**:
1. **Tải mô hình**: Từ file `.pkl` đã lưu
2. **Tiền xử lý**: Làm sạch văn bản đầu vào (giống như lúc huấn luyện)
3. **Vector hóa**: Chuyển văn bản thành vector số
4. **Dự đoán**: Sử dụng mô hình để phân loại
5. **Trả kết quả**: "Tích cực" hoặc "Tiêu cực"

**Ví dụ sử dụng**:
```python
result = predict_sentiment("This movie is fantastic!")
# Output: "Tích cực (Positive)"
```

### 3.5. File `app.py` - Ứng Dụng Web Streamlit

**Chức năng**: Tạo giao diện web để người dùng nhập đánh giá và xem kết quả

**Tính năng**:
- **Giao diện thân thiện**: Sử dụng Streamlit
- **Nhập liệu**: Text area để người dùng nhập đánh giá
- **Hiển thị kết quả**: 
  - ✅ Positive với màu xanh
  - ❌ Negative với màu đỏ
  - Kèm theo xác suất dự đoán

**Cách chạy**:
```bash
streamlit run app.py
```

### 3.6. File `requirements.txt` - Danh Sách Thư Viện

**Các thư viện chính**:
- `pandas`: Xử lý dữ liệu
- `numpy`: Tính toán số học
- `scikit-learn`: Machine Learning
- `joblib`: Lưu/tải mô hình
- `streamlit`: Tạo ứng dụng web
- `matplotlib`, `seaborn`: Vẽ biểu đồ

## 4. Pipeline Xử Lý Dữ Liệu

```
Dữ liệu thô (Raw Data)
        ↓
Tiền xử lý (Preprocessing)
    - Loại bỏ HTML tags
    - Chuyển về chữ thường  
    - Loại bỏ ký tự đặc biệt
        ↓
Chia dữ liệu (Train/Test Split)
    - 80% huấn luyện
    - 20% kiểm tra
        ↓
Vector hóa (TF-IDF)
    - Chuyển văn bản thành số
    - max_features=5000
    - ngram_range=(1,2)
        ↓
Huấn luyện mô hình (Logistic Regression)
        ↓
Đánh giá và lưu mô hình
        ↓
Ứng dụng dự đoán (Streamlit App)
```

## 5. Kỹ Thuật Sử Dụng

### 5.1. TF-IDF (Term Frequency - Inverse Document Frequency)
- **TF**: Tần suất xuất hiện của từ trong văn bản
- **IDF**: Mức độ hiếm của từ trong toàn bộ tập dữ liệu
- **Lợi ích**: Làm nổi bật các từ mang tính cảm xúc như "excellent", "terrible"

### 5.2. Logistic Regression
- **Ưu điểm**: Đơn giản, nhanh, hiệu quả với dữ liệu văn bản
- **Phù hợp**: Bài toán phân loại nhị phân (positive/negative)
- **Kết quả**: Cho ra xác suất thuộc từng lớp

## 6. Notebook Thử Nghiệm

### File `notebooks/experiments.ipynb`
- **Mục đích**: Thử nghiệm các tham số khác nhau
- **Nội dung**:
  - So sánh các giá trị `max_features`: 1000, 3000, 5000
  - Thử nghiệm `ngram_range`: (1,1) vs (1,2)
  - Điều chỉnh tham số `C` của Logistic Regression
- **Kết quả tốt nhất**: max_features=5000, ngram_range=(1,2), C=1.0

## 7. Cách Chạy Dự Án

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy pipeline huấn luyện
```bash
python src/preprocess.py      # Tiền xử lý dữ liệu
python src/split_data.py      # Chia dữ liệu
python src/train_model.py     # Huấn luyện mô hình
```

### Bước 3: Chạy ứng dụng web
```bash
streamlit run app.py
```

### Bước 4: Dự đoán từ command line
```bash
python src/predict.py
```

## 8. Ưu Điểm và Hạn Chế

### Ưu điểm:
- ✅ Pipeline hoàn chỉnh và có tổ chức
- ✅ Độ chính xác cao (~90%)
- ✅ Giao diện web thân thiện
- ✅ Code dễ hiểu và bảo trì

### Hạn chế:
- ❌ Không hiểu được ngữ cảnh phức tạp
- ❌ Phụ thuộc nhiều vào từ khóa
- ❌ Không xử lý được câu mỉa mai (sarcasm)
- ❌ Chỉ phân loại 2 lớp (positive/negative)

## 9. Kết Luận

Đây là một dự án Machine Learning cơ bản nhưng hoàn chỉnh, phù hợp để:
- Học cách xây dựng pipeline ML
- Hiểu về xử lý ngôn ngữ tự nhiên
- Thực hành với TF-IDF và Logistic Regression
- Tạo ứng dụng web đơn giản

Dự án có thể được mở rộng bằng cách:
- Sử dụng mô hình phức tạp hơn (LSTM, BERT)
- Thêm nhiều lớp cảm xúc (neutral, very positive, very negative)
- Cải thiện tiền xử lý dữ liệu
- Thêm tính năng phân tích chủ đề (topic modeling)