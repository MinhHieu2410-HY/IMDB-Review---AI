import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ----------------------- CÀI ĐẶT TRANG -----------------------
st.set_page_config(
    page_title="Phân tích cảm xúc Review Phim",
    page_icon="🎬",
    layout="centered"
)

# Tải stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Hàm làm sạch text
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))          # Xóa HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # Chỉ giữ chữ cái
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ----------------------- CACHE MÔ HÌNH -----------------------
@st.cache_resource
def load_model_and_vectorizer():
    # Tải dữ liệu IMDB trực tiếp từ GitHub (public)
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    df = pd.read_csv(url)
    
    # Tiền xử lý
    df['clean_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=40000,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['label']
    
    # Huấn luyện Logistic Regression (mô hình tốt nhất)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model, vectorizer

# Load mô hình (chỉ chạy 1 lần)
with st.spinner("Đang tải mô hình... (lần đầu hơi lâu tí nhé 🎥)"):
    model, vectorizer = load_model_and_vectorizer()

# ----------------------- GIAO DIỆN -----------------------
st.title("🎬 Phân tích cảm xúc Review Phim IMDB")
st.markdown("### Nhập review phim để xem nó **tích cực** hay **tiêu cực** nhé!")

user_input = st.text_area(
    "Nhập review của bạn ở đây:",
    height=150,
    placeholder="Ví dụ: This movie was absolutely fantastic! The acting was great and the story kept me on the edge of my seat..."
)

if st.button("Dự đoán cảm xúc", type="primary"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập một đoạn review để dự đoán!")
    else:
        with st.spinner("Đang phân tích..."):
            cleaned = clean_text(user_input)
            if cleaned == "":  # Nếu sau khi clean bị rỗng
                st.error("Review chỉ chứa ký tự đặc biệt hoặc stop words, không thể dự đoán được.")
            else:
                vec_input = vectorizer.transform([cleaned])
                prediction = model.predict(vec_input)[0]
                probability = model.predict_proba(vec_input)[0]
                
                pos_prob = probability[1]
                neg_prob = probability[0]
                
                if prediction == 1:
                    st.success("🎉 **Tích cực (Positive)**")
                    st.markdown(f"**Độ tin cậy:** {pos_prob:.1%} tích cực – {neg_prob:.1%} tiêu cực")
                else:
                    st.error("😢 **Tiêu cực (Negative)**")
                    st.markdown(f"**Độ tin cậy:** {neg_prob:.1%} tiêu cực – {pos_prob:.1%} tích cực")
                
                # Hiển thị review đã làm sạch (tùy chọn)
                with st.expander("Xem lại review sau khi làm sạch"):
                    st.text(cleaned)

# Footer
st.markdown("---")
st.caption("Dự án demo sử dụng mô hình Logistic Regression huấn luyện trên 50.000 review IMDB – Accuracy ~90%")
