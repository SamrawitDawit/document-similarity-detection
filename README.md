# Document Similarity Detection System

## 📌 Project Overview
This project is an **NLP-based Text Similarity Detection System** designed to identify duplicate or highly similar documents. It uses advanced Natural Language Processing (NLP) techniques to compute similarity scores and Machine Learning to classify pairs as duplicates or not.

The system is trained on the **Quora Customer Question Pairs** dataset using **TF-IDF Vectorization** and **Cosine Similarity** features fed into a **Logistic Regression** classifier.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
Ensure you have Python installed.

### 2. Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
The model must be trained before running the application. Open and run all cells in the Jupyter Notebook:
- File: `document_similarity.ipynb`
- This will generate the model files in the `models/` directory: `logistic_model.pkl` and `tfidf_vectorizer.pkl`.

### 4. Run the Application
Launch the user interface using Streamlit:
```bash
streamlit run app.py
```

---

## 🧭 Code Defense Guide (Line-by-Line Explanation)

### 📄 File: `document_similarity.ipynb` (Model Training)

#### 1. Importing Libraries
```python
import pandas as pd, numpy as np...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
```
*   **Defense**: "We import `pandas` for data manipulation, `nltk` for natural language processing (cleaning text), and `sklearn` for machine learning algorithms. specifically `TfidfVectorizer` for converting text to numbers and `LogisticRegression` for classification."

#### 2. Loading the Dataset
```python
dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class")
```
*   **Purpose**: Loads a massive dataset of question pairs labeled as duplicates (1) or not (0).
*   **Defense**: "We chose the Quora dataset because it provides a standard benchmark for semantic similarity, ensuring our model learns from high-quality real-world data."

#### 3. Preprocessing Function
```python
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)
```
*   **Purpose**: Cleans noise from the text.
*   **lower()**: Standardization (Apple == apple).
*   **remove punctuation**: Punctuation doesn't usually define similarity.
*   **lemmatize**: Converts "running" -> "run" so they match.
*   **Defense**: "Preprocessing reduces the vocabulary size and focuses the model on the core meaning of words, improving accuracy."

#### 4. Feature Extraction (TF-IDF)
```python
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
tfidf_vectorizer.fit(all_text)
```
*   **Purpose**: transform text into numerical vectors.
*   **TF-IDF**: Gives weight to rare but important words (unlike simple counts).
*   **Defense**: "We use TF-IDF instead of simple word counts because it downweights common words like 'the' and highlights unique, meaningful terms."

#### 5. Calculating Similarity (The Core Feature)
```python
similarity = cosine_similarity(tfidf_1[i], tfidf_2[i])[0][0]
```
*   **Purpose**: Measures the angle between two text vectors.
*   **Result**: A number between 0 (completely different) and 1 (identical).
*   **Defense**: "Cosine similarity is the standard metric for text distance. It measures orientation, not just magnitude, making it robust for texts of different lengths."

#### 6. Model Training (Logistic Regression)
```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```
*   **Purpose**: Learn the optimal threshold for similarity.
*   **Why Logistic Regression?**: It's simple, interpretable, and effective for binary classification (Duplicate / Not Duplicate).
*   **Defense**: "While we calculate cosine similarity, we use Logistic Regression to learn the optimal *threshold* for calling something a duplicate, rather than guessing a cutoff like 0.8."

#### 7. Evaluation
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```
*   **Defense**: "We evaluate using Accuracy, Precision, and Recall. Accuracy tells us overall correctness. Precision tells us: 'Of all the pairs we called duplicates, how many actually were?'"

---

### 📄 File: `app.py` (The User Interface)

*   **`preprocess_text`**: Must match the training preprocessing exactly.
*   **`vectorizer.transform`**: Converts user input to numbers using the *same dictionary* learned during training.
*   **`model.predict`**: The Logistic Regression model decides if the cosine similarity score is high enough to be a "Duplicate".

---

## ✅ Checking Accuracy of Results

1.  **Direct Match**: Input "Hello world" and "Hello world".
    *   *Result*: Cosine Similarity 1.0. Model Confidence ~99% and bigram difference.
2.  **Semantic Match**: Input "How do I turn on my phone?" and "What is the way to switch on my mobile?".
    *   *Result*: Similarity should be high because of shared meaningful words (stopwords removed).
3.  **Non-Match**: Input "I like cats" and "I hate dogs".
    *   *Result*: Similarity might be low (different words) or moderate (shared structure), but the *Model* should predict "Not Similar".
