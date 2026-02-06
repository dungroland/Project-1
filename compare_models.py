import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# BERT
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import os

# Thiết lập style cho plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_logistic_regression(X_train, y_train, X_test, y_test, sample_size=5000):
    """Train Logistic Regression model with fair comparison setup"""
    print("🔄 Training Logistic Regression...")
    start_time = time.time()
    
    # Use same sample size as BERT for fair comparison
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices]
        y_train_sample = y_train.iloc[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Use same test data size
    test_sample_size = min(1000, len(X_test))
    test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
    X_test_sample = X_test.iloc[test_indices]
    y_test_sample = y_test.iloc[test_indices]
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_sample)
    X_test_tfidf = tfidf.transform(X_test_sample)
    
    # Train model with more iterations for fair comparison
    lr_model = LogisticRegression(
        max_iter=10000,  # Increased iterations
        solver='lbfgs', 
        n_jobs=-1,
        C=1.0,  # Standard regularization
        random_state=42
    )
    lr_model.fit(X_train_tfidf, y_train_sample)
    
    # Predictions
    y_pred = lr_model.predict(X_test_tfidf)
    y_pred_proba = lr_model.predict_proba(X_test_tfidf)
    
    # Metrics
    accuracy = accuracy_score(y_test_sample, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_sample, y_pred, average='weighted')
    
    training_time = time.time() - start_time
    
    return {
        'model': lr_model,
        'vectorizer': tfidf,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': confusion_matrix(y_test_sample, y_pred),
        'test_labels': y_test_sample.values,
        'sample_size': len(X_train_sample)
    }

def train_bert_model(X_train, y_train, X_test, y_test, sample_size=5000):
    """Train BERT model with fair comparison setup"""
    print("🔄 Training DistilBERT...")
    start_time = time.time()
    
    # Use same sample size for fair comparison
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices]
        y_train_sample = y_train.iloc[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Use same test data size
    test_sample_size = min(1000, len(X_test))
    test_indices = np.random.choice(len(X_test), test_sample_size, replace=False)
    X_test_sample = X_test.iloc[test_indices]
    y_test_sample = y_test.iloc[test_indices]
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create datasets
    train_dataset = SentimentDataset(
        X_train_sample.values, y_train_sample.values, tokenizer
    )
    test_dataset = SentimentDataset(
        X_test_sample.values, y_test_sample.values, tokenizer
    )
    
    # Training arguments - 10 epochs for fair comparison
    training_args = TrainingArguments(
        output_dir='./models/bert_temp',
        num_train_epochs=10,  # Same as Logistic Regression iterations concept
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=2e-5,  # Standard BERT learning rate
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    training_time = time.time() - start_time
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': eval_results['eval_accuracy'],
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'f1': eval_results['eval_f1'],
        'training_time': training_time,
        'confusion_matrix': confusion_matrix(y_test_sample, y_pred),
        'test_labels': y_test_sample.values,
        'sample_size': len(X_train_sample)
    }

def create_comparison_plots(lr_results, bert_results):
    """Create comparison plots between models"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('So Sánh Mô Hình: Logistic Regression vs DistilBERT', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    models = ['Logistic Regression', 'DistilBERT']
    accuracies = [lr_results['accuracy'], bert_results['accuracy']]
    colors = ['#3498db', '#e74c3c']
    
    bars = axes[0,0].bar(models, accuracies, color=colors, alpha=0.8)
    axes[0,0].set_title('Độ Chính Xác (Accuracy)', fontweight='bold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. All Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lr_scores = [lr_results['accuracy'], lr_results['precision'], lr_results['recall'], lr_results['f1']]
    bert_scores = [bert_results['accuracy'], bert_results['precision'], bert_results['recall'], bert_results['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0,1].bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#3498db', alpha=0.8)
    axes[0,1].bar(x + width/2, bert_scores, width, label='DistilBERT', color='#e74c3c', alpha=0.8)
    
    axes[0,1].set_title('So Sánh Tất Cả Metrics', fontweight='bold')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(metrics)
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1)
    
    # 3. Training Time Comparison
    times = [lr_results['training_time'], bert_results['training_time']]
    bars = axes[0,2].bar(models, times, color=colors, alpha=0.8)
    axes[0,2].set_title('Thời Gian Huấn Luyện (giây)', fontweight='bold')
    axes[0,2].set_ylabel('Thời gian (s)')
    
    for bar, time_val in zip(bars, times):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01, 
                      f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. Confusion Matrix - Logistic Regression
    sns.heatmap(lr_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                ax=axes[1,0])
    axes[1,0].set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
    axes[1,0].set_ylabel('True Label')
    axes[1,0].set_xlabel('Predicted Label')
    
    # 5. Confusion Matrix - BERT
    sns.heatmap(bert_results['confusion_matrix'], annot=True, fmt='d', cmap='Reds',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                ax=axes[1,1])
    axes[1,1].set_title('Confusion Matrix - DistilBERT', fontweight='bold')
    axes[1,1].set_ylabel('True Label')
    axes[1,1].set_xlabel('Predicted Label')
    
    # 6. Model Comparison Summary
    axes[1,2].axis('off')
    summary_text = f"""
    📊 SETUP SO SÁNH CÔNG BẰNG
    
    🔧 Cấu hình:
    • Cùng dữ liệu: {lr_results['sample_size']} samples
    • BERT: 10 epochs, lr=2e-5
    • LogReg: 10k iterations, C=1.0
    
    🏆 Độ chính xác cao nhất:
    {'DistilBERT' if bert_results['accuracy'] > lr_results['accuracy'] else 'Logistic Regression'}
    ({max(bert_results['accuracy'], lr_results['accuracy']):.3f})
    
    ⚡ Huấn luyện nhanh nhất:
    {'Logistic Regression' if lr_results['training_time'] < bert_results['training_time'] else 'DistilBERT'}
    ({min(lr_results['training_time'], bert_results['training_time']):.1f}s)
    
    📈 F1-Score cao nhất:
    {'DistilBERT' if bert_results['f1'] > lr_results['f1'] else 'Logistic Regression'}
    ({max(bert_results['f1'], lr_results['f1']):.3f})
    
    💡 Khuyến nghị:
    {'DistilBERT cho độ chính xác cao' if bert_results['accuracy'] > lr_results['accuracy'] else 'Logistic Regression cho tốc độ nhanh'}
    """
    
    axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Bắt đầu so sánh mô hình Logistic Regression vs DistilBERT")
    print("=" * 60)
    
    # Load data
    print("📊 Đang tải dữ liệu...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"✅ Dữ liệu train: {len(train_df)} samples")
    print(f"✅ Dữ liệu test: {len(test_df)} samples")
    
    X_train, y_train = train_df['review'], train_df['sentiment']
    X_test, y_test = test_df['review'], test_df['sentiment']
    
    # Train Logistic Regression
    print("\n" + "="*60)
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test, sample_size=5000)
    print(f"✅ Logistic Regression - Accuracy: {lr_results['accuracy']:.3f}")
    print(f"📊 Sample size: {lr_results['sample_size']} samples")
    print(f"⏱️  Training time: {lr_results['training_time']:.1f}s")
    
    # Train BERT (with same sample size for fair comparison)
    print("\n" + "="*60)
    bert_results = train_bert_model(X_train, y_train, X_test, y_test, sample_size=5000)
    print(f"✅ DistilBERT - Accuracy: {bert_results['accuracy']:.3f}")
    print(f"📊 Sample size: {bert_results['sample_size']} samples")
    print(f"⏱️  Training time: {bert_results['training_time']:.1f}s")
    
    # Create comparison plots
    print("\n" + "="*60)
    print("📈 Đang tạo biểu đồ so sánh...")
    create_comparison_plots(lr_results, bert_results)
    
    # Print detailed results
    print("\n" + "="*60)
    print("📋 KẾT QUẢ CHI TIẾT:")
    print("\n🔵 LOGISTIC REGRESSION:")
    print(f"   Accuracy:  {lr_results['accuracy']:.4f}")
    print(f"   Precision: {lr_results['precision']:.4f}")
    print(f"   Recall:    {lr_results['recall']:.4f}")
    print(f"   F1-Score:  {lr_results['f1']:.4f}")
    print(f"   Time:      {lr_results['training_time']:.1f}s")
    
    print("\n🔴 DISTILBERT:")
    print(f"   Accuracy:  {bert_results['accuracy']:.4f}")
    print(f"   Precision: {bert_results['precision']:.4f}")
    print(f"   Recall:    {bert_results['recall']:.4f}")
    print(f"   F1-Score:  {bert_results['f1']:.4f}")
    print(f"   Time:      {bert_results['training_time']:.1f}s")
    
    # Winner
    print("\n🏆 KẾT LUẬN:")
    if bert_results['accuracy'] > lr_results['accuracy']:
        print(f"   DistilBERT thắng với độ chính xác cao hơn {bert_results['accuracy'] - lr_results['accuracy']:.3f}")
    else:
        print(f"   Logistic Regression thắng với độ chính xác cao hơn {lr_results['accuracy'] - bert_results['accuracy']:.3f}")
    
    print(f"   Logistic Regression nhanh hơn {bert_results['training_time'] - lr_results['training_time']:.1f}s")
    
    print("\n✅ Hoàn thành! Biểu đồ đã được lưu vào 'model_comparison.png'")

if __name__ == "__main__":
    main()