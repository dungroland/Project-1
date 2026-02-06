import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

class BERTSentimentPredictor:
    def __init__(self, model_path="models/distilbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {self.device}")
        
        # Tải tokenizer và model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping nhãn
        self.label_map = {0: "Tiêu cực (Negative)", 1: "Tích cực (Positive)"}
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản (tương tự như lúc training)"""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)  # Loại bỏ HTML
        text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Loại bỏ ký tự đặc biệt
        text = re.sub(r"\s+", " ", text).strip()  # Loại bỏ khoảng trắng thừa
        return text
    
    def predict(self, text):
        """Dự đoán cảm xúc cho một văn bản"""
        # Tiền xử lý
        clean_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Chuyển sang device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'prediction': self.label_map[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'negative': predictions[0][0].item(),
                'positive': predictions[0][1].item()
            }
        }
    
    def predict_batch(self, texts):
        """Dự đoán cho nhiều văn bản cùng lúc"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    """Demo sử dụng BERT predictor"""
    try:
        # Khởi tạo predictor
        predictor = BERTSentimentPredictor()
        
        print("=== BERT Sentiment Analysis Demo ===")
        print("Nhập 'exit' để thoát\n")
        
        while True:
            user_input = input("Nhập đánh giá phim: ")
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.strip() == "":
                print("Vui lòng nhập nội dung!")
                continue
            
            # Dự đoán
            result = predictor.predict(user_input)
            
            # Hiển thị kết quả
            print(f"\n🎯 Kết quả: {result['prediction']}")
            print(f"📊 Độ tin cậy: {result['confidence']:.2%}")
            print(f"📈 Chi tiết:")
            print(f"   • Negative: {result['probabilities']['negative']:.2%}")
            print(f"   • Positive: {result['probabilities']['positive']:.2%}")
            print("-" * 50)
    
    except FileNotFoundError:
        print("❌ Không tìm thấy mô hình BERT!")
        print("Vui lòng chạy 'python src/train_bert_model.py' trước.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()