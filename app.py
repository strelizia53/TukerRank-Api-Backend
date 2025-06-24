from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# App setup
app = Flask(__name__)
CORS(app)

# Firebase init
cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load RoBERTa sentiment model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Elo constant
K = 32

# RoBERTa-based sentiment analysis
def analyze_sentiment_roberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[scores.argmax()]
    return sentiment, float(scores.max())

# Elo logic
def calculate_elo(current_rating, sentiment, rating_out_of_5):
    expected_score = 0.5
    sentiment_score = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}.get(sentiment, 0.5)
    result_score = (sentiment_score + (rating_out_of_5 / 5)) / 2
    new_rating = current_rating + K * (result_score - expected_score)
    return round(new_rating)

@app.route("/", methods=["GET"])
def home():
    return "TukeRank API (RoBERTa powered) is running 🚀", 200

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    username = data.get("username")
    review = data.get("review", "")
    rating = data.get("rating", 3)

    if not username:
        return jsonify({"error": "Missing username"}), 400

    try:
        sentiment, confidence = analyze_sentiment_roberta(review)

        users_ref = db.collection("users").where("username", "==", username).limit(1).stream()
        user_doc = next(users_ref, None)

        if not user_doc:
            return jsonify({"error": "User not found"}), 404

        user_data = user_doc.to_dict()
        current_elo = user_data.get("elo", 1000)
        new_elo = calculate_elo(current_elo, sentiment, rating)
        elo_change = new_elo - current_elo

        db.collection("users").document(user_doc.id).update({"elo": new_elo})

        db.collection("feedbacks").add({
            "driverId": username,
            "review": review,
            "rating": rating,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "eloChange": elo_change
        })

        return jsonify({
            "message": "Feedback processed",
            "sentiment": sentiment,
            "confidence": confidence,
            "newElo": new_elo,
            "eloChange": elo_change
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feedback/<username>", methods=["GET"])
def get_user_feedbacks(username):
    try:
        feedbacks_ref = db.collection("feedbacks").where("driverId", "==", username)
        feedbacks = feedbacks_ref.stream()
        results = [fb.to_dict() for fb in feedbacks]
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/feedbacks", methods=["GET"])
def get_all_feedbacks():
    try:
        sentiment = request.args.get("sentiment")
        search = request.args.get("search", "").lower()
        feedbacks_ref = db.collection("feedbacks")
        feedbacks = feedbacks_ref.stream()
        result = []
        for fb in feedbacks:
            data = fb.to_dict()
            if sentiment and data.get("sentiment") != sentiment:
                continue
            if search and search not in data.get("review", "").lower():
                continue
            result.append(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
    