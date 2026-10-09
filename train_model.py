"""
Train and evaluate both models:
1. Sentiment: Positive / Negative / Neutral
2. Emotion: Happy / Sad / Angry / Frustrated / Fearful / Surprised /
            Love / Disappointed / Excited / Neutral

Both models use the same understandable pipeline:
clean text -> TF-IDF -> Logistic Regression
"""

from pathlib import Path
import json
import re
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

SENTIMENT_DATA = DATASET_DIR / "sentiment_data.csv"
EMOTION_DATA = DATASET_DIR / "emotion_data.csv"

SENTIMENT_MODEL = MODEL_DIR / "sentiment_model.pkl"
SENTIMENT_VECTORIZER = MODEL_DIR / "tfidf_vectorizer.pkl"
EMOTION_MODEL = MODEL_DIR / "emotion_model.pkl"
EMOTION_VECTORIZER = MODEL_DIR / "emotion_tfidf_vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

FALLBACK_STOP_WORDS = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'hers', 'him', 'his', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'me', 'my', 'of', 'on', 'or', 'our', 'ours', 'she', 'so', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your', 'yours']

def get_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Offline-safe fallback: the project still uses NLTK when its
        # corpus is available, but does not crash without an internet download.
        return set(FALLBACK_STOP_WORDS)

STOP_WORDS = get_stop_words()

def clean_text(text):
    """Lowercase, remove URLs/special characters, tokenize, remove stop words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

def train_classifier(data_path, label_column, model_path, vectorizer_path, random_state=42):
    df = pd.read_csv(data_path)
    if not {"text", label_column}.issubset(df.columns):
        raise ValueError(f"{data_path.name} must contain 'text' and '{label_column}' columns.")

    df = df.dropna(subset=["text", label_column]).copy()
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df[label_column],
        test_size=0.20,
        random_state=random_state,
        stratify=df[label_column]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "labels": labels,
        "confusion_matrix": cm.tolist()
    }

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer, metrics, y_test, y_pred

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("Training sentiment model...")
    _, _, sentiment_metrics, sentiment_y, sentiment_pred = train_classifier(
        SENTIMENT_DATA, "sentiment", SENTIMENT_MODEL, SENTIMENT_VECTORIZER
    )

    print("Training emotion model...")
    emotion_model, _, emotion_metrics, emotion_y, emotion_pred = train_classifier(
        EMOTION_DATA, "emotion", EMOTION_MODEL, EMOTION_VECTORIZER
    )

    all_metrics = {
        "sentiment": sentiment_metrics,
        "emotion": emotion_metrics
    }
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print("\n===== SENTIMENT MODEL =====")
    print(f"Accuracy : {sentiment_metrics['accuracy']:.2%}")
    print(f"Precision: {sentiment_metrics['precision_weighted']:.2%}")
    print(f"Recall   : {sentiment_metrics['recall_weighted']:.2%}")
    print(f"F1-score : {sentiment_metrics['f1_weighted']:.2%}")
    print(classification_report(sentiment_y, sentiment_pred, zero_division=0))

    print("\n===== EMOTION MODEL =====")
    print(f"Classes  : {', '.join(emotion_model.classes_)}")
    print(f"Accuracy : {emotion_metrics['accuracy']:.2%}")
    print(f"Precision: {emotion_metrics['precision_weighted']:.2%}")
    print(f"Recall   : {emotion_metrics['recall_weighted']:.2%}")
    print(f"F1-score : {emotion_metrics['f1_weighted']:.2%}")
    print(classification_report(emotion_y, emotion_pred, zero_division=0))

    print("\nSaved files:")
    for path in [SENTIMENT_MODEL, SENTIMENT_VECTORIZER, EMOTION_MODEL, EMOTION_VECTORIZER, METRICS_PATH]:
        print(f" - {path}")

if __name__ == "__main__":
    main()
