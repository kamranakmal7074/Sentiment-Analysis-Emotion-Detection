from pathlib import Path
import json
import re
import joblib
import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
SENTIMENT_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.pkl"
EMOTION_VECTORIZER_PATH = MODEL_DIR / "emotion_tfidf_vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

app = Flask(__name__)

required_files = [
    SENTIMENT_MODEL_PATH,
    SENTIMENT_VECTORIZER_PATH,
    EMOTION_MODEL_PATH,
    EMOTION_VECTORIZER_PATH,
]
missing = [str(path.name) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Model files are missing: "
        + ", ".join(missing)
        + ". Run 'python train_model.py' first."
    )

# Load both independent models once when Flask starts.
sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
sentiment_vectorizer = joblib.load(SENTIMENT_VECTORIZER_PATH)
emotion_model = joblib.load(EMOTION_MODEL_PATH)
emotion_vectorizer = joblib.load(EMOTION_VECTORIZER_PATH)

FALLBACK_STOP_WORDS = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'hers', 'him', 'his', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'me', 'my', 'of', 'on', 'or', 'our', 'ours', 'she', 'so', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your', 'yours']

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # Offline-safe fallback for a first-time/local college demo.
    STOP_WORDS = set(FALLBACK_STOP_WORDS)

SENTIMENT_EMOJIS = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}

EMOTION_EMOJIS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "frustrated": "😤",
    "fearful": "😨",
    "surprised": "😲",
    "love": "❤️",
    "disappointed": "😔",
    "excited": "🤩",
    "neutral": "😐",
}

EMOTION_ORDER = [
    "happy", "sad", "angry", "frustrated", "fearful",
    "surprised", "love", "disappointed", "excited", "neutral"
]

def clean_text(text):
    """Keep preprocessing identical to the training pipeline."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

def predict_with_model(text, model, vectorizer):
    cleaned = clean_text(text)
    model_input = cleaned if cleaned.strip() else str(text).lower().strip()

    features = vectorizer.transform([model_input])
    prediction = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    probability_map = {
        str(label): round(float(prob), 4)
        for label, prob in zip(model.classes_, probabilities)
    }
    confidence = round(probability_map[prediction] * 100, 2)
    return prediction, probability_map, confidence

def analyze_text(text):
    sentiment, sentiment_probs, sentiment_confidence = predict_with_model(
        text, sentiment_model, sentiment_vectorizer
    )
    emotion, emotion_probs, emotion_confidence = predict_with_model(
        text, emotion_model, emotion_vectorizer
    )

    # Return every expected emotion to keep the chart stable.
    emotion_percentages = {}
    for label in EMOTION_ORDER:
        emotion_percentages[label] = round(emotion_probs.get(label, 0.0) * 100, 2)

    sentiment_percentages = {
        "positive": round(sentiment_probs.get("positive", 0.0) * 100, 2),
        "neutral": round(sentiment_probs.get("neutral", 0.0) * 100, 2),
        "negative": round(sentiment_probs.get("negative", 0.0) * 100, 2),
    }

    return {
        "text": text,
        "sentiment": sentiment,
        "display_sentiment": sentiment.capitalize(),
        "sentiment_emoji": SENTIMENT_EMOJIS.get(sentiment, "😐"),
        "sentiment_confidence": sentiment_confidence,
        "sentiment_probabilities": sentiment_percentages,

        "emotion": emotion,
        "display_emotion": emotion.capitalize(),
        "emotion_emoji": EMOTION_EMOJIS.get(emotion, "😐"),
        "emotion_confidence": emotion_confidence,
        "emotion_probabilities": emotion_percentages,
    }

@app.route("/")
def index():
    metrics = {}
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics = {}
    return render_template("index.html", metrics=metrics)

@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({"success": False, "error": "Invalid text input."}), 400

    text = text.strip()
    if not text:
        return jsonify({"success": False, "error": "Please enter some text first."}), 400

    if len(text) > 5000:
        return jsonify({
            "success": False,
            "error": "Please keep your text within 5000 characters."
        }), 400

    try:
        result = analyze_text(text)
        result["success"] = True
        return jsonify(result)
    except Exception:
        app.logger.exception("Sentiment and emotion analysis failed")
        return jsonify({
            "success": False,
            "error": "Sorry, the text could not be analyzed. Please try again."
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
