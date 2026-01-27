import streamlit as st
import nltk
import string
import joblib
import docx
import PyPDF2

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Document Similarity Detection",
    page_icon="📄",
    layout="centered"
)


# --------------------------------------------------
# NLTK Setup
# --------------------------------------------------
@st.cache_resource
def download_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk()

# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/logistic_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf_vectorizer = load_models()

# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# File Readers
# --------------------------------------------------
def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def read_docx(file):
    document = docx.Document(file)
    return " ".join([p.text for p in document.paragraphs])

def extract_text(file):
    if file.type == "text/plain":
        return read_txt(file)
    elif file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    else:
        return ""

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📄 Document Similarity Detection")
st.markdown(
    "This application uses **TF-IDF + Cosine Similarity + Logistic Regression** "
    "to determine whether two documents are similar."
)

input_method = st.radio(
    "Choose Input Method:",
    ["✍️ Paste Text", "📂 Upload Documents"],
    horizontal=True
)

text1, text2 = "", ""

if input_method == "✍️ Paste Text":
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("First Text", height=250)
    with col2:
        text2 = st.text_area("Second Text", height=250)

else:
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader(
            "Upload First Document",
            type=["txt", "pdf", "docx"]
        )
        if file1:
            text1 = extract_text(file1)

    with col2:
        file2 = st.file_uploader(
            "Upload Second Document",
            type=["txt", "pdf", "docx"]
        )
        if file2:
            text2 = extract_text(file2)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")

if st.button("🔍 Check Similarity", use_container_width=True):

    if not text1.strip() or not text2.strip():
        st.warning("⚠️ Please provide both documents or texts.")
    else:
        with st.spinner("Analyzing documents..."):

            # Preprocess
            p1 = preprocess_text(text1)
            p2 = preprocess_text(text2)

            # Vectorize using trained TF-IDF
            tfidf_1 = tfidf_vectorizer.transform([p1])
            tfidf_2 = tfidf_vectorizer.transform([p2])

            # Cosine similarity
            similarity_score = cosine_similarity(tfidf_1, tfidf_2)[0][0]

            # Model prediction
            prediction = model.predict([[similarity_score]])[0]
            confidence = model.predict_proba([[similarity_score]])[0][prediction]

        # Results
        st.subheader("📊 Results")
        col1, col2 = st.columns(2)
        col1.metric("Cosine Similarity", f"{similarity_score:.4f}")
        col2.metric("Model Confidence", f"{confidence*100:.2f}%")

        if prediction == 1:
            st.success("🟢 Documents are SIMILAR (Model Prediction)")
        else:
            st.error("🔴 Documents are NOT SIMILAR (Model Prediction)")
