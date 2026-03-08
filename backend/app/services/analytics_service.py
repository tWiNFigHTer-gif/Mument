from app.services.json_storage import read_reviews

def get_review_statistics():
    reviews = read_reviews()

    total_reviews = len(reviews)

    if total_reviews == 0:
        return {
            "total_reviews": 0,
            "average_rating": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }

    total_rating = sum(review["rating"] for review in reviews)
    average_rating = total_rating / total_reviews

    positive = sum(1 for r in reviews if r.get("sentiment") == "positive")
    neutral = sum(1 for r in reviews if r.get("sentiment") == "neutral")
    negative = sum(1 for r in reviews if r.get("sentiment") == "negative")

    return {
        "total_reviews": total_reviews,
        "average_rating": round(average_rating, 2),
        "positive": positive,
        "neutral": neutral,
        "negative": negative
    }