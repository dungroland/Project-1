import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng device: {device}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize văn bản
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

def main():
    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Lấy mẫu nhỏ để test nhanh (có thể bỏ dòng này để train full data)
    train_df = train_df.sample(n=5000, random_state=42)
    test_df = test_df.sample(n=1000, random_state=42)
    
    # 2. Khởi tạo tokenizer và model
    print("Đang tải DistilBERT...")
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # 3. Tạo datasets
    print("Đang chuẩn bị datasets...")
    train_dataset = SentimentDataset(
        train_df['review'].values,
        train_df['sentiment'].values,
        tokenizer
    )
    
    test_dataset = SentimentDataset(
        test_df['review'].values,
        test_df['sentiment'].values,
        tokenizer
    )
    
    # 4. Cấu hình training
    training_args = TrainingArguments(
        output_dir='./models/distilbert_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # 5. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 6. Bắt đầu training
    print("Bắt đầu huấn luyện DistilBERT...")
    trainer.train()
    
    # 7. Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    eval_results = trainer.evaluate()
    print(f"Kết quả đánh giá: {eval_results}")
    
    # 8. Lưu mô hình
    print("Đang lưu mô hình...")
    os.makedirs("models/distilbert", exist_ok=True)
    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    
    print("Hoàn thành huấn luyện DistilBERT!")
    print(f"Độ chính xác: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()