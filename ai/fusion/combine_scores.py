# Weighted merge logic

from ai.unsupervised_clustering.clustering import infer_cluster_sentiment

CONFIDENCE_THRESHOLD = 0.85

def fuse_predictions(distilbert_result: dict, review_text: str) -> dict:
    """
    if distilbert confidence >= 85% ath mathi,
    athil korav aanel kmeans vech confirm chyynm
    """
    confidence = distilbert_result["score"]

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "label": distilbert_result["label"],
            "confidence": round(confidence, 4),
            "decided_by": "distilbert"
        }
    else:
        kmeans_result = infer_cluster_sentiment(review_text)
        return {
            "label": kmeans_result["label"],
            "confidence": round(confidence, 4),
            "decided_by": "kmeans",
            "cluster_id": kmeans_result["cluster_id"]
        }


