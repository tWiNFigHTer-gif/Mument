# Clustering script
import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews_clean.json")
OUTPUT_REVIEWS_PATH = os.path.join(BASE_DIR, "data", "artifacts", "clustered_reviews.json")
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, "data", "artifacts", "kmeans_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "artifacts", "tfidf_vectorizer.pkl")
OUTPUT_META_PATH = os.path.join(BASE_DIR, "data", "artifacts", "cluster_metadata.json")

# Map cluster no to sentiment label like i did in sentiment analysis
CLUSTER_SENTIMENT_MAP = {
    0: "positive",
    1: "negative",
    2: "neutral",
    3: "neutral",
    4: "negative"
}

# ---------- Load Data ----------
with open(INPUT_PATH, "r") as f:
    reviews = json.load(f)

texts = [r["clean_text"] for r in reviews]

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5
)

X = vectorizer.fit_transform(texts)

# ---------- KMeans ----------
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X)

for i, r in enumerate(reviews):
    r["cluster_id"] = int(clusters[i])

# ---------- Extract Top Keywords ----------
feature_names = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

cluster_metadata = {}

for i in range(k):
    top_terms = [feature_names[ind] for ind in order_centroids[i, :10]]
    cluster_metadata[i] = top_terms

# ---------- Save Outputs ----------
with open(OUTPUT_REVIEWS_PATH, "w") as f:
    json.dump(reviews, f, indent=2)

with open(OUTPUT_META_PATH, "w") as f:
    json.dump(cluster_metadata, f, indent=2)

with open(KMEANS_MODEL_PATH, "wb") as f:
    pickle.dump(kmeans, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Clustering completed successfully.")

def infer_cluster_sentiment(text: str) -> dict:
    with open(KMEANS_MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform([text])
    cluster_id = int(kmeans.predict(X)[0])
    label = CLUSTER_SENTIMENT_MAP.get(cluster_id, "NEUTRAL")

    return{"label": label, "cluster_id": cluster_id}

