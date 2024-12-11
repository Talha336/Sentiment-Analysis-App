from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Analyze sentiment
    result = sentiment_pipeline(text)
    sentiment = result[0]["label"]
    score = result[0]["score"]

    # Add an emoji based on sentiment
    emoji = "😊" if sentiment == "POSITIVE" else "😞"

    return jsonify({"sentiment": sentiment, "score": score, "emoji": emoji})

if __name__ == '__main__':
    # Ensure the app runs on any host (useful for development)
    app.run()
