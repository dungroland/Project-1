import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import os

# Thiết lập style cho plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def train_and_evaluate_logistic():
    """Train và đánh giá mô hình Logistic Regression"""
    print("🚀 Bắt đầu huấn luyện Logistic Regression")
    print("=" * 60)
    
    # 1. Load dữ liệu
    print("📊 Đang tải dữ liệu...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"✅ Dữ liệu train: {len(train_df)} samples")
    print(f"✅ Dữ liệu test: {len(test_df)} samples")
    print(f"📈 Phân bố train - Positive: {sum(train_df['sentiment'])}, Negative: {len(train_df) - sum(train_df['sentiment'])}")
    print(f"📈 Phân bố test - Positive: {sum(test_df['sentiment'])}, Negative: {len(test_df) - sum(test_df['sentiment'])}")
    
    X_train, y_train = train_df['review'], train_df['sentiment']
    X_test, y_test = test_df['review'], test_df['sentiment']
    
    # 2. TF-IDF Vectorization
    print("\n🔄 Đang thực hiện TF-IDF vectorization...")
    start_time = time.time()
    
    tfidf = TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        strip_accents='ascii'
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train.values.astype('U'))
    X_test_tfidf = tfidf.transform(X_test.values.astype('U'))
    
    vectorization_time = time.time() - start_time
    print(f"✅ TF-IDF hoàn thành trong {vectorization_time:.2f}s")
    print(f"📊 Kích thước ma trận train: {X_train_tfidf.shape}")
    print(f"📊 Kích thước ma trận test: {X_test_tfidf.shape}")
    
    # 3. Train mô hình
    print("\n🔄 Đang huấn luyện Logistic Regression...")
    train_start = time.time()
    
    lr_model = LogisticRegression(
        max_iter=10000,
        solver='lbfgs',
        n_jobs=-1,
        C=1.0,
        random_state=42
    )
    
    lr_model.fit(X_train_tfidf, y_train)
    training_time = time.time() - train_start
    
    print(f"✅ Huấn luyện hoàn thành trong {training_time:.2f}s")
    
    # 4. Dự đoán và đánh giá
    print("\n🔄 Đang đánh giá mô hình...")
    eval_start = time.time()
    
    y_pred = lr_model.predict(X_test_tfidf)
    y_pred_proba = lr_model.predict_proba(X_test_tfidf)
    
    eval_time = time.time() - eval_start
    
    # 5. Tính toán metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. In kết quả
    print("\n" + "="*60)
    print("📋 KẾT QUẢ LOGISTIC REGRESSION:")
    print("="*60)
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🎯 F1-Score:  {f1:.4f}")
    print(f"⏱️  Thời gian vectorization: {vectorization_time:.2f}s")
    print(f"⏱️  Thời gian training: {training_time:.2f}s")
    print(f"⏱️  Thời gian evaluation: {eval_time:.2f}s")
    print(f"⏱️  Tổng thời gian: {vectorization_time + training_time + eval_time:.2f}s")
    
    # 7. Classification Report
    print("\n📊 CLASSIFICATION REPORT:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # 8. Confusion Matrix
    print("\n📊 CONFUSION MATRIX:")
    print("-" * 60)
    print("Predicted ->  Negative  Positive")
    print(f"Negative      {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"Positive      {cm[1,0]:8d}  {cm[1,1]:8d}")
    
    # 9. Lưu mô hình
    print("\n💾 Đang lưu mô hình...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, "models/sentiment_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("✅ Đã lưu mô hình vào thư mục models/")
    
    # 10. Tạo biểu đồ
    create_logistic_plots(accuracy, precision, recall, f1, cm, y_pred_proba, y_test)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'training_time': training_time + vectorization_time,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def create_logistic_plots(accuracy, precision, recall, f1, cm, y_pred_proba, y_test):
    """Tạo các biểu đồ cho Logistic Regression"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Kết Quả Logistic Regression - Sentiment Analysis', fontsize=16, fontweight='bold')
    
    # 1. Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = axes[0,0].bar(metrics, values, color=colors, alpha=0.8)
    axes[0,0].set_title('Các Chỉ Số Đánh Giá', fontweight='bold')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_ylim(0, 1)
    
    # Thêm giá trị lên thanh
    for bar, val in zip(bars, values):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'],
                ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix', fontweight='bold')
    axes[0,1].set_ylabel('True Label')
    axes[0,1].set_xlabel('Predicted Label')
    
    # 3. Prediction Confidence Distribution
    positive_probs = y_pred_proba[y_test == 1, 1]  # Prob of positive for true positives
    negative_probs = y_pred_proba[y_test == 0, 0]  # Prob of negative for true negatives
    
    axes[0,2].hist(positive_probs, bins=20, alpha=0.7, label='True Positive', color='green')
    axes[0,2].hist(negative_probs, bins=20, alpha=0.7, label='True Negative', color='red')
    axes[0,2].set_title('Phân Bố Độ Tin Cậy Dự Đoán', fontweight='bold')
    axes[0,2].set_xlabel('Confidence Score')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()
    
    # 4. ROC-like curve (Precision-Recall)
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    
    axes[1,0].plot(recall_curve, precision_curve, color='blue', linewidth=2)
    axes[1,0].set_title('Precision-Recall Curve', fontweight='bold')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Feature Importance (Top words)
    feature_names = np.array(joblib.load("models/tfidf_vectorizer.pkl").get_feature_names_out())
    lr_model = joblib.load("models/sentiment_model.pkl")
    
    # Top positive words
    top_positive_idx = np.argsort(lr_model.coef_[0])[-10:]
    top_positive_words = feature_names[top_positive_idx]
    top_positive_scores = lr_model.coef_[0][top_positive_idx]
    
    axes[1,1].barh(range(len(top_positive_words)), top_positive_scores, color='green', alpha=0.7)
    axes[1,1].set_yticks(range(len(top_positive_words)))
    axes[1,1].set_yticklabels(top_positive_words)
    axes[1,1].set_title('Top 10 Positive Words', fontweight='bold')
    axes[1,1].set_xlabel('Coefficient Value')
    
    # 6. Model Summary
    axes[1,2].axis('off')
    summary_text = f"""
    📊 TỔNG KẾT MÔ HÌNH
    
    🎯 Độ chính xác: {accuracy:.1%}
    📈 F1-Score: {f1:.3f}
    
    🔧 Cấu hình:
    • TF-IDF: 5000 features
    • N-grams: (1,2)
    • Solver: lbfgs
    • Max iterations: 10,000
    
    📊 Dữ liệu:
    • Train: {len(joblib.load("models/tfidf_vectorizer.pkl").vocabulary_)} từ vựng
    • Test samples: {len(y_test)}
    
    ✅ Mô hình đã sẵn sàng sử dụng!
    """
    
    axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Biểu đồ đã được lưu vào 'logistic_regression_results.png'")

if __name__ == "__main__":
    results = train_and_evaluate_logistic()
    print("\n🎉 Hoàn thành huấn luyện Logistic Regression!")