Sentiment Classification with TF-IDF & Logistic Regression 

1. Mục tiêu dự án 

Dự án này nhằm xây dựng một hệ thống Phân loại cảm xúc (Sentiment Classification) cho các đánh giá phim, với 2 nhãn cảm xúc: 

Positive (Tích cực) 

Negative (Tiêu cực) 

Mục tiêu chính 

Làm quen với pipeline Machine Learning hoàn chỉnh cho bài toán xử lý ngôn ngữ tự nhiên (NLP) 

Áp dụng TF-IDF để trích xuất đặc trưng từ dữ liệu văn bản 

Huấn luyện mô hình Logistic Regression cho bài toán phân loại 

Đánh giá hiệu năng mô hình bằng các metrics chuẩn (Accuracy) 

Xây dựng chương trình dự đoán cảm xúc cho dữ liệu văn bản mới 

2. Dataset 

Tên dataset: IMDB Movie Reviews 

Nguồn: Kaggle 

Số lượng: 50,000 đánh giá phim 

Loại dữ liệu: Văn bản (Text) 

Dataset bao gồm các đánh giá phim thực tế, do đó chứa nhiều yếu tố nhiễu như: 

Thẻ HTML (<br />) 

Ký tự đặc biệt 

Cách viết không thống nhất (chữ hoa – chữ thường) 

Các câu mang sắc thái trung lập hoặc mơ hồ 

3. Công nghệ sử dụng 

3.1. Ngôn ngữ Python3 

3.2. Thư viện 

pandas 

numpy 

scikit-learn 

joblib 

3.3. Kỹ thuật Machine Learning 

TF-IDF (Feature Extraction) 

Logistic Regression (One-vs-Rest) 

4. Cấu trúc thư mục dự án 

sentiment-classification/ 
├── data/ 
│   ├── raw/                # Dữ liệu gốc từ Kaggle 
│   └── processed/          # Dữ liệu đã làm sạch & chia tập 
│ 
├── models/                 # Lưu model và vectorizer 
│   ├── sentiment_model.pkl 
│   └── tfidf_vectorizer.pkl 
│ 
├── src/ 
│   ├── preprocess.py       # Tiền xử lý văn bản 
│   ├── split_data.py       # Chia train / test 
│   ├── train_model.py      # Huấn luyện & đánh giá 
│   └── predict.py          # Dự đoán cảm xúc 
│ 
├── requirements.txt 
└── README.md 

5. Pipeline tổng thể của hệ thống 

Raw Text 
  ↓ 
Text Cleaning (HTML, Regex) 
  ↓ 
Normalization (Lowercase) 
  ↓ 
TF-IDF Vectorization 
  ↓ 
Logistic Regression 
  ↓ 
Sentiment Output (Positive / Negative) 

6. Tiền xử lý dữ liệu 

6.1. Mô hình ngôn ngữ và tiền xử lý văn bản 

Tiền xử lý văn bản là bước quan trọng nhằm giảm nhiễu và nâng cao chất lượng dữ liệu đầu vào. 
Các kỹ thuật được sử dụng bao gồm: 

Text Cleaning: Loại bỏ ký tự đặc biệt, HTML tags, và các thành phần không cần thiết 

Regular Expression (Regex): Chuẩn hóa văn bản bằng các biểu thức chính quy 

Lowercasing: Chuyển toàn bộ văn bản về chữ thường để tránh phân biệt từ do khác nhau về kiểu chữ 

Stop Words Removal: Loại bỏ các từ phổ biến không mang nhiều ý nghĩa cảm xúc như the, is, and 

6.2. Chuẩn hóa nhãn cảm xúc 

negative → 0 

positive → 1 

6.3. Pipeline tiền xử lý 

Raw Text → Cleaning → Normalization → Cleaned Dataset 

7. Chia dữ liệu Train / Test: src/split_data.py 

Train set: 80% 

Test set: 20% 

random_state = 42 

Sử dụng stratify theo nhãn cảm xúc để đảm bảo phân bố dữ liệu cân bằng 

Cleaned Dataset → Train (80%) + Test (20%) 

8. Huấn luyện mô hình: src/train_model.py 

8.1. Kỹ thuật Vector hóa văn bản – TF-IDF 

TF-IDF (Term Frequency – Inverse Document Frequency) là kỹ thuật biểu diễn văn bản phản ánh mức độ quan trọng của một từ trong một văn bản so với toàn bộ tập dữ liệu. 

TF (Term Frequency): Tần suất xuất hiện của từ trong văn bản 

IDF (Inverse Document Frequency): Mức độ hiếm của từ trong toàn bộ tập dữ liệu 

Lợi ích của TF-IDF trong phân loại cảm xúc 

Làm nổi bật các từ mang sắc thái cảm xúc như excellent, terrible, boring 

Giảm ảnh hưởng của các từ xuất hiện phổ biến 

Phù hợp với các mô hình phân loại tuyến tính như Logistic Regression 

8.2. Cấu hình mô hình 

TF-IDF Vectorizer 

max_features = 5000 

ngram_range = (1, 2) (unigram + bigram) 

Logistic Regression 

multi_class = 'ovr' 

max_iter = 1000 

Solver: lbfgs 

8.3. Kiến trúc mô hình 

Text → TF-IDF → Logistic Regression → {Negative | Positive} 

9. Lưu trữ mô hình 

Sau khi huấn luyện, mô hình và vectorizer được lưu tại thư mục models/: 

models/ 
├── sentiment_model.pkl 
└── tfidf_vectorizer.pkl 

Sử dụng joblib giúp tái sử dụng mô hình cho inference mà không cần huấn luyện lại. 

10. Dự đoán cảm xúc (Inference): src/predict.py 

Quy trình suy luận 

User Input → Preprocess → TF-IDF → Model → Sentiment 

12. Hạn chế của mô hình 

Không nắm bắt được ngữ cảnh dài hạn 

Phụ thuộc nhiều vào từ khóa 