import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import joblib
import os

# Tải NLTK data (chạy lần đầu)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Hàm làm sạch text
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))  # Xóa HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Giữ chữ cái
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Cache tải và xử lý data
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    df = pd.read_csv(url)
    df['cleanreview'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

# Cache vectorizer và train/test split
@st.cache_resource
def get_vectorizer_and_data():
    df = load_and_preprocess_data()
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8, stop_words='english')
    X = vectorizer.fit_transform(df['cleanreview'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return vectorizer, X_train, X_test, y_train, y_test

# Cache huấn luyện models
@st.cache_resource
def train_models():
    vectorizer, X_train, X_test, y_train, y_test = get_vectorizer_and_data()
    
    # Logistic Regression
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    
    # SVM
    model_svm = LinearSVC()
    model_svm.fit(X_train, y_train)
    
    # Decision Tree
    model_dt = DecisionTreeClassifier(max_depth=50, min_samples_split=10, random_state=42)
    model_dt.fit(X_train, y_train)
    
    return model_lr, model_svm, model_dt, X_test, y_test

# Tạo word clouds
@st.cache_data
def generate_wordclouds(df):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8, stop_words='english')
    X = vectorizer.fit_transform(df['cleanreview'])
    features = vectorizer.get_feature_names_out()
    
    pos_idx = df[df['label'] == 1].index
    neg_idx = df[df['label'] == 0].index
    
    pos_score = X[pos_idx].mean(axis=0).A1
    neg_score = X[neg_idx].mean(axis=0).A1
    
    pos_words = {features[i]: pos_score[i] for i in range(len(features)) if pos_score[i] > neg_score[i]}
    neg_words = {features[i]: neg_score[i] for i in range(len(features)) if neg_score[i] > pos_score[i]}
    
    wc_pos = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pos_words)
    wc_neg = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(neg_words)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(wc_pos)
    ax[0].set_title("Positive Reviews")
    ax[0].axis("off")
    ax[1].imshow(wc_neg)
    ax[1].set_title("Negative Reviews")
    ax[1].axis("off")
    
    return fig, sorted(pos_words.items(), key=lambda x: x[1], reverse=True)[:10], sorted(neg_words.items(), key=lambda x: x[1], reverse=True)[:10]

# Hàm đánh giá model
def evaluate_model(model_name, model, X_test, y_test):
    if model_name == "SVM":
        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)
    else:
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    
    return report, cm, acc, auc

# Giao diện Streamlit
st.set_page_config(page_title="IMDB Sentiment Analysis Demo", layout="wide")

st.title("IMDB Movie Review Sentiment Analysis Demo")
st.markdown("Dự án phân tích cảm xúc review phim từ dataset IMDB (50k reviews). Sử dụng TF-IDF và các mô hình ML để dự đoán positive/negative.")

# Phần 1: Phân tích dữ liệu
st.header("Phân tích Dữ liệu")
df = load_and_preprocess_data()
st.write(f"Dữ liệu: {df.shape[0]} reviews (balanced: 25k positive, 25k negative)")

wc_fig, top_pos, top_neg = generate_wordclouds(df)
st.pyplot(wc_fig)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Positive Words")
    st.write(pd.DataFrame(top_pos, columns=["Word", "Score"]))
with col2:
    st.subheader("Top 10 Negative Words")
    st.write(pd.DataFrame(top_neg, columns=["Word", "Score"]))

# Phần 2: Kết quả mô hình
st.header("Kết quả Mô hình Machine Learning")
model_lr, model_svm, model_dt, X_test, y_test = train_models()

models = {
    "Logistic Regression": model_lr,
    "SVM": model_svm,
    "Decision Tree": model_dt
}

results = {}
for name, model in models.items():
    report, cm, acc, auc = evaluate_model(name, model, X_test, y_test)
    results[name] = {"Accuracy": acc, "AUC": auc, "Report": report, "CM": cm}

# Bảng so sánh
st.subheader("So sánh Accuracy và AUC")
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["Accuracy"] for m in results],
    "AUC": [results[m]["AUC"] for m in results]
})
st.table(comparison_df)

# Chi tiết từng model
for name in models:
    with st.expander(f"Chi tiết {name}"):
        st.write("Classification Report:")
        st.write(pd.DataFrame(results[name]["Report"]).T)
        
        st.write("Confusion Matrix:")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(results[name]["CM"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

# Phần 3: Dự đoán
st.header("Dự đoán Sentiment cho Review Mới")
user_input = st.text_area("Nhập review phim:")
if st.button("Dự đoán"):
    if user_input:
        vectorizer, _, _, _, _ = get_vectorizer_and_data()
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        pred = model_lr.predict(vec_input)[0]  # Dùng LR vì tốt nhất
        prob = model_lr.predict_proba(vec_input)[0][1]
        
        sentiment = "Positive" if pred == 1 else "Negative"
        st.success(f"Kết quả: {sentiment} (Xác suất positive: {prob:.2f})")
    else:
        st.warning("Vui lòng nhập review!")

st.markdown("---")
st.caption("Demo by Grok - Dựa trên IMDB Dataset. Deploy trên Streamlit Sharing từ GitHub.")
