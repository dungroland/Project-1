# 📋 Tóm Tắt File .gitignore

## ✅ Đã Cập Nhật

File `.gitignore` đã được cập nhật để phù hợp với dự án Sentiment Classification.

## 📂 Các Thư Mục/File Được Ignore

### 1. **Python & Environment**
- `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`
- `venv/`, `.venv/`, `env/`, `ENV/`

### 2. **Data Files** (Quan trọng!)
- `data/raw/*.csv` - Dataset gốc (thường rất lớn)
- `data/processed/*.csv` - Dữ liệu đã xử lý
- Giữ lại cấu trúc: `!data/raw/.gitkeep`, `!data/processed/.gitkeep`

### 3. **Models** (Quan trọng!)
- `models/*.pkl` - Mô hình Logistic Regression
- `models/*.joblib` - Vectorizer
- `models/distilbert/` - Mô hình BERT
- `models/bert_temp/` - Checkpoint tạm
- Giữ lại cấu trúc: `!models/.gitkeep`

### 4. **Generated Files**
- `*.png`, `*.jpg`, `*.pdf` - Biểu đồ
- `tfidf_weights.csv` - Kết quả phân tích
- `results/` - Thư mục kết quả
- Ngoại trừ: `!docs/*.png`, `!images/*.png`

### 5. **Jupyter Notebooks**
- `.ipynb_checkpoints/`
- `*.nbconvert.ipynb`

### 6. **Logs & Experiments**
- `logs/`, `*.log`
- `mlruns/` (MLflow)

### 7. **OS & Editor**
- `.DS_Store` (macOS)
- `Thumbs.db` (Windows)
- `.vscode/`, `.idea/`

### 8. **Streamlit**
- `.streamlit/secrets.toml`

### 9. **Environment Variables**
- `.env`, `.env.*`

### 10. **Testing & Coverage**
- `.coverage`, `htmlcov/`, `.pytest_cache/`

### 11. **Build & Distribution**
- `build/`, `dist/`, `*.egg-info/`

## 🎯 Lý Do Ignore

### ❌ Không nên commit:
1. **Dataset lớn** (50,000 reviews) - Quá nặng cho Git
2. **Models đã train** - File .pkl có thể vài trăm MB
3. **Biểu đồ tạm** - Có thể tạo lại bất cứ lúc nào
4. **Virtual environment** - Mỗi người có môi trường riêng
5. **Cache & logs** - Không cần thiết

### ✅ Nên commit:
1. **Source code** (`.py`)
2. **Requirements** (`requirements.txt`)
3. **README & Documentation**
4. **Notebooks** (`.ipynb`)
5. **Config files**
6. **Cấu trúc thư mục** (`.gitkeep`)

## 📝 Cách Sử Dụng

### Nếu muốn commit một file đã bị ignore:
```bash
git add -f path/to/file
```

### Kiểm tra file nào sẽ bị ignore:
```bash
git status --ignored
```

### Xóa cache và apply .gitignore mới:
```bash
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

## 🔧 Tùy Chỉnh

Nếu bạn muốn:
- **Commit models nhỏ**: Xóa dòng `models/*.pkl`
- **Commit biểu đồ quan trọng**: Thêm `!important_chart.png`
- **Commit sample data**: Thêm `!data/sample/*.csv`

## ✅ Kết Luận

File `.gitignore` hiện tại đã đủ cho dự án này và tuân thủ best practices:
- Không commit file lớn
- Không commit file tạm
- Giữ lại cấu trúc dự án
- Dễ dàng tái tạo môi trường