# Clustering script
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews_clean.json")
OUTPUT_REVIEWS_PATH = os.path.join(BASE_DIR, "data", "artifacts", "clustered_reviews.json")
OUTPUT_META_PATH = os.path.join(BASE_DIR, "data", "artifacts", "cluster_metadata.json")

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

print("Clustering completed successfully.")
