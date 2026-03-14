from app.services.json_storage import read_reviews

def get_review_statistics():
    raw_reviews = read_reviews()
    reviews = [review for review in raw_reviews if isinstance(review, dict)]

    total_reviews = len(reviews)

    if total_reviews == 0:
        return {
            "total_reviews": 0,
            "average_rating": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }

    def _safe_rating(review: dict) -> int:
        try:
            return max(1, min(5, int(review.get("rating", 0))))
        except (TypeError, ValueError):
            return 3

    total_rating = sum(_safe_rating(review) for review in reviews)
    average_rating = total_rating / total_reviews

    def _sentiment_label(review: dict) -> str:
        label = review.get("sentiment")
        return label.lower() if isinstance(label, str) else "neutral"

    positive = sum(1 for r in reviews if _sentiment_label(r) == "positive")
    neutral = sum(1 for r in reviews if _sentiment_label(r) == "neutral")
    negative = sum(1 for r in reviews if _sentiment_label(r) == "negative")

    return {
        "total_reviews": total_reviews,
        "average_rating": round(average_rating, 2),
        "positive": positive,
        "neutral": neutral,
        "negative": negative
    }