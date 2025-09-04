import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -----------------Preprocessing Function-----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(doc: str) -> str:
    if not isinstance(doc, str):
        return ""
    doc = doc.lower()
    doc = re.sub(r'<[^>]+>', ' ', doc)
    doc = re.sub(r'[^a-z\s]', ' ', doc)
    tokens = word_tokenize(doc)
    processed = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(processed)