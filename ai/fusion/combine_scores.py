# Weighted merge logic
import json
from pathlib import Path
from collections import defaultdict

# ---------- Base Path ----------
BASE_DIR = Path(__file__).resolve().parents[2]

SENTIMENT_PATH = BASE_DIR / "data" / "artifacts" / "sentiment_scores.json"
CLUSTER_PATH = BASE_DIR / "data" / "artifacts" / "clustered_reviews.json"
OUTPUT_PATH = BASE_DIR / "data" / "artifacts" / "fusion_outputs.json"

# ---------- Load Data ----------
with open(SENTIMENT_PATH, "r") as f:
    sentiment_reviews = json.load(f)

with open(CLUSTER_PATH, "r") as f:
    cluster_reviews = json.load(f)

# ---------- Create Cluster Map ----------
cluster_map = {
    r["review_id"]: r["cluster_id"]
    for r in cluster_reviews
}

# ---------- Merge Cluster IDs ----------
for r in sentiment_reviews:
    r["cluster_id"] = cluster_map.get(r["review_id"], None)

# ---------- Compute Cluster-Level Sentiment Average ----------
cluster_scores = defaultdict(list)

for r in sentiment_reviews:
    cluster_scores[r["cluster_id"]].append(r["svm_score"])

cluster_avg = {
    cid: sum(scores) / len(scores)
    for cid, scores in cluster_scores.items()
}

# ---------- Fusion Score ----------
for r in sentiment_reviews:
    cluster_signal = cluster_avg[r["cluster_id"]]
    fusion_score = 0.7 * r["svm_score"] + 0.3 * cluster_signal
    r["fusion_score"] = fusion_score

# ---------- Save Output ----------
with open(OUTPUT_PATH, "w") as f:
    json.dump(sentiment_reviews, f, indent=2)

print("Fusion completed successfully.")

