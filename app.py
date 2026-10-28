
from pathlib import Path
import os

import joblib
import serpapi
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "truthlens_model.pkl"

# Load the existing trained AI model
model = joblib.load(MODEL_PATH)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthLens | Fake News Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #101827;
            color: #f8fafc;
            margin: 0;
            padding: 40px 16px;
        }
        .container {
            max-width: 800px;
            margin: 20px auto;
            background: #1e293b;
            padding: 30px;
            border-radius: 18px;
        }
        h1 { color: #67e8f9; }
        h2 { color: #a5f3fc; }
        p { color: #cbd5e1; line-height: 1.6; }
        textarea {
            box-sizing: border-box;
            width: 100%;
            height: 180px;
            padding: 14px;
            border-radius: 10px;
            font-size: 16px;
            margin: 12px 0;
        }
        button {
            background: #0891b2;
            color: white;
            border: 0;
            border-radius: 8px;
            padding: 13px 22px;
            cursor: pointer;
            font-size: 16px;
        }
        .result, .article {
            margin-top: 20px;
            padding: 18px;
            background: #334155;
            border-radius: 10px;
        }
        .article a {
            color: #67e8f9;
            overflow-wrap: anywhere;
        }
        .note { font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔎 TruthLens</h1>
        <p>AI-powered news analysis and evidence search</p>
        <p>Paste a news headline or article below.</p>

        <form method="POST">
            <textarea name="news"
                placeholder="Paste news text here..."
                required>{{ news }}</textarea>
            <button type="submit">Analyze News</button>
        </form>

        {% if result %}
        <div class="result">
            <h2>AI Prediction: {{ result }}</h2>
            {% if confidence is not none %}
                <p>Model confidence: {{ confidence }}%</p>
            {% endif %}
            <p>{{ explanation }}</p>
        </div>
        {% endif %}

        {% if search_message %}
        <div class="result">
            <h2>Related News Search</h2>
            <p>{{ search_message }}</p>
        </div>
        {% endif %}

        {% for article in articles %}
        <div class="article">
            <h3>
                <a href="{{ article.link }}" target="_blank"
                   rel="noopener noreferrer">
                    {{ article.title }}
                </a>
            </h3>
            <p>{{ article.snippet }}</p>
            <p><a href="{{ article.link }}" target="_blank"
                  rel="noopener noreferrer">Read article</a></p>
        </div>
        {% endfor %}

        <p class="note">
            Important: The AI prediction is not a verified fact-check.
            Search results provide leads for further investigation;
            they do not automatically prove a claim is true or false.
            Check trustworthy, independent sources before sharing news.
        </p>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    news = ""
    result = None
    confidence = None
    explanation = ""
    articles = []
    search_message = ""

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if news:
            # Existing AI prediction
            try:
                prediction = model.predict([news])[0]
                label = str(prediction).strip().lower()

                if label in ("fake", "0"):
                    result = "Potentially Fake"
                elif label in ("real", "true", "1"):
                    result = "Likely Real"
                else:
                    result = f"Model prediction: {prediction}"

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([news])[0]
                    confidence = round(
                        float(max(probabilities)) * 100, 2
                    )

                explanation = (
                    "This prediction reflects patterns learned during "
                    "training. Verify the claim independently."
                )
            except Exception:
                result = "Prediction Error"
                explanation = (
                    "The AI model could not analyze this text. "
                    "Check the model file and terminal error."
                )

            # Search related news using SerpApi
            api_key = os.getenv("SERPAPI_KEY")

            if not api_key:
                search_message = (
                    "SerpApi API key is not configured yet. "
                    "Set the SERPAPI_KEY environment variable "
                    "and restart the application."
                )
            else:
                try:
                    client = serpapi.Client(
                        api_key=api_key,
                        timeout=20
                    )
                    search_results = client.search(
                        engine="google",
                        q=news[:300]
                    )

                    for item in search_results.get(
                        "organic_results", []
                    )[:5]:
                        link = item.get("link", "")
                        title = item.get("title", "")

                        # Only show valid web links
                        if (link.startswith("https://")
                                or link.startswith("http://")):
                            articles.append({
                                "title": title or "Untitled result",
                                "link": link,
                                "snippet": item.get(
                                    "snippet",
                                    "No summary provided."
                                )
                            })

                    if articles:
                        search_message = (
                            "Search results found. Open the articles "
                            "and compare their evidence with the claim."
                        )
                    else:
                        search_message = (
                            "The search returned no article results."
                        )

                except Exception as error:
                    app.logger.exception("SerpApi search failed")
                    search_message = (
                        "The news search could not be completed. "
                        "Check your API key, account verification, "
                        "search quota, internet connection, and "
                        "the terminal error."
                    )

    return render_template_string(
        HTML,
        news=news,
        result=result,
        confidence=confidence,
        explanation=explanation,
        articles=articles,
        search_message=search_message,
    )


if __name__ == "__main__":
    app.run(debug=True)