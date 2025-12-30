import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------- CÀI ĐẶT TRANG -----------------------
st.set_page_config(
    page_title="Phân tích cảm xúc Review Phim",
    page_icon="🎬",
    layout="centered"
)

# Stopwords - giữ "not" để xử lý phủ định
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

# Hàm clean text - đã fix negation
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    text = text.lower()
    
    # Mở rộng contraction
    contractions = {
        "don't": "do not", "doesn't": "does not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "won't": "will not", "wouldn't": "would not", "can't": "cannot",
        "couldn't": "could not", "shouldn't": "should not"
    }
    for contr, full in contractions.items():
        text = text.replace(contr, full)
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Rule-based boost: từ điển từ tích cực / tiêu cực mạnh
POSITIVE_KEYWORDS = {'like', 'love', 'great', 'good', 'amazing', 'best', 'excellent', 'wonderful', 'fantastic', 'awesome', 'brilliant', 'enjoy', 'perfect', 'favorite'}
NEGATIVE_KEYWORDS = {'hate', 'worst', 'terrible', 'awful', 'bad', 'horrible', 'boring', 'waste', 'disappointing', 'poor', 'stupid', 'dull'}

def boost_sentiment(cleaned_text, original_prob_positive):
    words = set(cleaned_text.split())
    
    pos_count = len(words & POSITIVE_KEYWORDS)
    neg_count = len(words & NEGATIVE_KEYWORDS)
    
    # Mỗi từ positive +15%, negative -15%, giới hạn 0-1
    boost = (pos_count - neg_count) * 0.15
    new_prob = original_prob_positive + boost
    new_prob = max(0.0, min(1.0, new_prob))  # clamp giữa 0 và 1
    
    return new_prob

# ----------------------- LOAD MODEL -----------------------
@st.cache_resource
def load_model_and_vectorizer():
    with st.spinner("Đang tải dữ liệu và huấn luyện mô hình... (lần đầu ~1-2 phút)"):
        url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
        df = pd.read_csv(url)
        
        df['clean_review'] = df['review'].apply(clean_text)
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
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
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ----------------------- GIAO DIỆN -----------------------
st.title("🎬 Phân tích cảm xúc Review Phim IMDB")

user_input = st.text_area(
    "Nhập review phim:",
    height=150,
    placeholder="I like this movie..."
)

if st.button("Dự đoán cảm xúc", type="primary"):
    if not user_input.strip():
        st.warning("Hãy nhập một đoạn review để dự đoán!")
    else:
        with st.spinner("Đang phân tích..."):
            cleaned = clean_text(user_input)
            if not cleaned.strip():
                st.error("Review sau xử lý bị rỗng. Hãy thử viết dài hơn.")
            else:
                vec_input = vectorizer.transform([cleaned])
                
                # Dự đoán gốc từ model
                prob_positive = model.predict_proba(vec_input)[0][1]
                
                # Áp dụng rule-based boost
                boosted_prob_positive = boost_sentiment(cleaned, prob_positive)
                
                prob_negative = 1 - boosted_prob_positive
                
                if boosted_prob_positive >= 0.5:
                    st.success(f"🎉 **Tích cực (Positive)**")
                    st.markdown(f"**Độ tin cậy:** {boosted_prob_positive:.1%} tích cực – {prob_negative:.1%} tiêu cực")
                else:
                    st.error(f"😢 **Tiêu cực (Negative)**")
                    st.markdown(f"**Độ tin cậy:** {prob_negative:.1%} tiêu cực – {boosted_prob_positive:.1%} tích cực")
                
