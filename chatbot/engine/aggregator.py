# Cluster + sentiment summaries
import json
from pathlib import Path
from collections import Counter, defaultdict

# ---------- Base Path ----------
BASE_DIR = Path(__file__).resolve().parents[2]

FUSION_PATH = BASE_DIR / "data" / "artifacts" / "fusion_outputs.json"
CLUSTER_META_PATH = BASE_DIR / "data" / "artifacts" / "cluster_metadata.json"
OUTPUT_PATH = BASE_DIR / "data" / "artifacts" / "chatbot_summary.json"

# ---------- Load Data ----------
with open(FUSION_PATH, "r") as f:
    reviews = json.load(f)

with open(CLUSTER_META_PATH, "r") as f:
    cluster_metadata = json.load(f)

total_reviews = len(reviews)

# ---------- Sentiment Distribution ----------
sentiment_counts = Counter(r["svm_label"] for r in reviews)
negative_ratio = sentiment_counts.get("negative", 0) / total_reviews

# ---------- Cluster Distribution ----------
cluster_counts = Counter(r["cluster_id"] for r in reviews)

# ---------- Average Fusion Score Per Cluster ----------
cluster_scores = defaultdict(list)

for r in reviews:
    cluster_scores[r["cluster_id"]].append(r["fusion_score"])

cluster_avg = {
    cid: sum(scores) / len(scores)
    for cid, scores in cluster_scores.items()
}

# ---------- Find Most Problematic Cluster ----------
worst_cluster = min(cluster_avg, key=cluster_avg.get)

summary = {
    "total_reviews": total_reviews,
    "sentiment_distribution": dict(sentiment_counts),
    "negative_ratio": negative_ratio,
    "cluster_distribution": dict(cluster_counts),
    "worst_cluster_id": worst_cluster,
    "worst_cluster_keywords": cluster_metadata[str(worst_cluster)],
    "worst_cluster_score": cluster_avg[worst_cluster]
}

# ---------- Save ----------
with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("Chatbot summary generated successfully.")
