
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FAKE_FILE = os.path.join(DATA_DIR, "Fake.csv")
REAL_FILE = os.path.join(DATA_DIR, "True.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "truthlens_model.pkl")

def main():
    # Check that both datasets exist
    if not os.path.exists(FAKE_FILE):
        raise FileNotFoundError(f"Dataset not found: {FAKE_FILE}")

    if not os.path.exists(REAL_FILE):
        raise FileNotFoundError(f"Dataset not found: {REAL_FILE}")

    print("Loading datasets...")

    fake = pd.read_csv(FAKE_FILE)
    real = pd.read_csv(REAL_FILE)

    # Assign labels: 0 = Fake, 1 = Real
    fake["label"] = 0
    real["label"] = 1

    # Combine title and article text when available
    for df in (fake, real):
        if "text" not in df.columns:
            raise ValueError("A dataset is missing the required 'text' column.")

        if "title" in df.columns:
            df["content"] = (
                df["title"].fillna("").astype(str)
                + " "
                + df["text"].fillna("").astype(str)
            )
        else:
            df["content"] = df["text"].fillna("").astype(str)

    data = pd.concat(
        [fake[["content", "label"]], real[["content", "label"]]],
        ignore_index=True
    )

    # Remove empty and duplicate articles
    data["content"] = data["content"].str.strip()
    data = data[data["content"].str.len() > 20]
    data = data.drop_duplicates(subset=["content"])

    if data["label"].nunique() < 2:
        raise ValueError("Both fake and real news examples are required.")

    print(f"Total articles: {len(data)}")
    print("Splitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    # TF-IDF converts text into numerical features.
    # Logistic Regression learns to distinguish the dataset labels.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("Training the machine-learning model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"]
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved successfully to: {MODEL_FILE}")
    print("TruthLens model training completed!")

if __name__ == "__main__":
    main()