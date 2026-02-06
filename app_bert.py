import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import plotly.express as px
import pandas as pd

# Cấu hình trang
st.set_page_config(
    page_title="BERT Sentiment Analysis", 
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_bert_model():
    """Tải mô hình BERT (cache để tránh tải lại)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert")
        model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except:
        return None, None, None

def preprocess_text(text):
    """Tiền xử lý văn bản"""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text, tokenizer, model, device):
    """Dự đoán cảm xúc"""
    clean_text = preprocess_text(text)
    
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {
            'negative': predictions[0][0].item(),
            'positive': predictions[0][1].item()
        }
    }

# UI chính
def main():
    # Header
    st.title("🤖 BERT Sentiment Analysis")
    st.markdown("**Phân loại cảm xúc sử dụng DistilBERT**")
    st.markdown("---")
    
    # Tải mô hình
    tokenizer, model, device = load_bert_model()
    
    if tokenizer is None:
        st.error("❌ Không thể tải mô hình BERT!")
        st.info("Vui lòng chạy `python src/train_bert_model.py` để huấn luyện mô hình trước.")
        return
    
    # Hiển thị thông tin mô hình
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "DistilBERT")
    with col2:
        st.metric("Device", str(device).upper())
    with col3:
        st.metric("Max Length", "512 tokens")
    
    st.markdown("---")
    
    # Input area
    st.subheader("📝 Nhập đánh giá của bạn:")
    user_input = st.text_area(
        "Đánh giá phim:",
        height=150,
        placeholder="Ví dụ: This movie is absolutely fantastic! The acting was superb and the plot was engaging..."
    )
    
    # Predict button
    if st.button("🔍 Phân tích cảm xúc", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Vui lòng nhập nội dung đánh giá!")
        else:
            # Hiển thị loading
            with st.spinner("🤖 BERT đang phân tích..."):
                result = predict_sentiment(user_input, tokenizer, model, device)
            
            # Hiển thị kết quả
            st.markdown("---")
            st.subheader("📊 Kết quả phân tích:")
            
            # Kết quả chính
            if result['prediction'] == 1:
                st.success(f"✅ **POSITIVE** (Tích cực)")
                st.success(f"🎯 Độ tin cậy: **{result['confidence']:.1%}**")
            else:
                st.error(f"❌ **NEGATIVE** (Tiêu cực)")
                st.error(f"🎯 Độ tin cậy: **{result['confidence']:.1%}**")
            
            # Biểu đồ xác suất
            st.subheader("📈 Phân bố xác suất:")
            
            prob_data = pd.DataFrame({
                'Cảm xúc': ['Negative', 'Positive'],
                'Xác suất': [
                    result['probabilities']['negative'],
                    result['probabilities']['positive']
                ],
                'Màu': ['#ff4444', '#44ff44']
            })
            
            fig = px.bar(
                prob_data, 
                x='Cảm xúc', 
                y='Xác suất',
                color='Cảm xúc',
                color_discrete_map={'Negative': '#ff4444', 'Positive': '#44ff44'},
                title="Xác suất dự đoán"
            )
            fig.update_layout(showlegend=False, height=400)
            fig.update_yaxis(tickformat='.1%')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Chi tiết số liệu
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Negative", 
                    f"{result['probabilities']['negative']:.1%}",
                    delta=f"{result['probabilities']['negative'] - 0.5:.1%}"
                )
            with col2:
                st.metric(
                    "Positive", 
                    f"{result['probabilities']['positive']:.1%}",
                    delta=f"{result['probabilities']['positive'] - 0.5:.1%}"
                )
    
    # Sidebar với thông tin
    with st.sidebar:
        st.header("ℹ️ Thông tin")
        st.markdown("""
        **Mô hình:** DistilBERT
        
        **Ưu điểm:**
        - Hiểu ngữ cảnh sâu
        - Xử lý câu phức tạp
        - Độ chính xác cao (~93-95%)
        
        **So với Logistic Regression:**
        - Chính xác hơn
        - Hiểu mỉa mai
        - Xử lý phủ định tốt hơn
        """)
        
        st.markdown("---")
        st.markdown("**🚀 Được tạo bởi BERT & Streamlit**")

if __name__ == "__main__":
    main()