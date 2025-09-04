# sentiment_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from text_preprocess import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import movie_reviews
# ---------- NLTK downloads (first run will download)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.path.append('./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('movie_reviews')

# ---------- Load sample dataset (NLTK movie_reviews) if user doesn't upload
@st.cache_data
def load_nltk_sample():
    docs = []
    labels = []
    for fid in movie_reviews.fileids():
        docs.append(movie_reviews.raw(fid))
        labels.append(movie_reviews.categories(fid)[0])  # 'pos' / 'neg'
    df = pd.DataFrame({'review': docs, 'sentiment': labels})
    df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})
    return df

# ---------- Streamlit UI
st.set_page_config(page_title="Sentiment Analysis 📝", layout="wide")
st.title("Public Reviews Sentiment Analysis")

sidebar = st.sidebar
sidebar.header("Options")

use_sample = sidebar.checkbox("Use sample dataset (NLTK movie_reviews)", value=True)
uploaded_file = sidebar.file_uploader("Or upload CSV (columns: review, sentiment)", type=['csv'])
sidebar.subheader("Select Model")
model_choice = sidebar.radio("Model Type:", ["Naive Bayes (fast)", "Logistic Regression (accurate)"], index = None)
save_model_btn = sidebar.button("Save last trained model")

# Load dataset
if use_sample or uploaded_file is None:
    df = load_nltk_sample()
    st.info("Using NLTK movie_reviews sample dataset (2000 reviews).")
    st.subheader("(Demo) Dataset preview")
    st.write(df.sample(5))
    st.write("(Demo) Dataset Class distribution:")
    st.bar_chart(df['sentiment'].value_counts())
else:
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='cp1252')
            st.subheader(f"{uploaded_file.name} Dataset preview")
            st.write(df.sample(5))
            st.write(f"{uploaded_file.name} Dataset Class distribution:")
            st.bar_chart(df['sentiment'].value_counts())
            st.write("Total File Rows:", len(df))
        else:
            st.error("CSV must contain 'review' and 'sentiment' columns.")
    except Exception as e:
        st.error(f"Couldn't read uploaded file: {e}")

# Train-test split
st.subheader("Perform a Review Test to Train a Model")
if model_choice is None:
    st.info("Select a model from the Sidebar to train it")
test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
random_state = 42

if st.button("Train model"):
    X = df['review'].astype(str)
    y = df['sentiment'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=random_state, stratify=y)

    # Build pipeline
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text, max_features=10000, ngram_range=(1,2))
    try:
        if model_choice.startswith("Naive"):
            clf = MultinomialNB()
        else:
            clf = LogisticRegression(max_iter=1000, solver='liblinear')
    except NameError as e:
        st.info("Please Select your Model to Train!")
    except AttributeError as e:
        st.info("Please Select your Model to Train!")

    pipeline = Pipeline([('tfidf', vectorizer), ('clf', clf)])

    with st.spinner("Training model..."):
        model = pipeline.fit(X_train, y_train)
    st.success("Training completed.")

    # Evaluate
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    st.metric("Test accuracy", f"{acc:.4f}")
    st.subheader("Classification report")
    report = classification_report(y_test, preds, output_dict=True)
    st.table(pd.DataFrame(report).T)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Saving to session state for later use
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test

# Single prediction UI
st.subheader("Try a custom review")
input_review = st.text_area("Type a review to predict sentiment (positive = 1, negative = 0):", height=120)
if st.button("Predict review"):
    if 'model' not in st.session_state:
        st.error("Train the model first.")
    else:
        model = st.session_state['model']
        pred = model.predict([input_review])[0]
        st.write("Predicted label:", int(pred))
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([input_review])[0]
            st.write("Probabilities:", p.round(3))

# Show some test examples with predictions
if 'model' in st.session_state and st.checkbox("Show sample test predictions"):
    m = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    sample_df = pd.DataFrame({'review': X_test.sample(10 if len(df)>=10 else 2, random_state=1), 'actual': y_test.sample(10 if len(df)>=10 else 2, random_state=1)})
    sample_df['predicted'] = sample_df['review'].apply(lambda r: m.predict([r])[0])
    st.write(sample_df)

# Save model button
if save_model_btn:
    if 'model' in st.session_state:
        joblib.dump(st.session_state['model'], "sentiment_pipeline.joblib")
        st.success("Saved model to sentiment_pipeline.joblib")
    else:
        st.error("Train a model first to save it.")
