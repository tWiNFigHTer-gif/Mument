from fastapi import APIRouter, Query
from app.schemas.review_schema import Review, ReviewAnalysisRequest
from app.services.json_storage import read_reviews, write_reviews
from app.services.sentiment_service import analyze_review_text
import uuid
from datetime import datetime

router = APIRouter()


def _normalize_review_record(record: dict) -> dict:
    username = record.get("username") or record.get("name") or "Guest"
    comment = record.get("comment") or ""
    try:
        rating = int(record.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0

    sentiment = record.get("sentiment")
    sentiment = sentiment.lower() if isinstance(sentiment, str) else None

    return {
        "id": record.get("id") or str(uuid.uuid4()),
        "username": username,
        "rating": max(1, min(5, rating)) if rating else 3,
        "comment": comment,
        "sentiment": sentiment,
        "car_key": record.get("car_key"),
        "timestamp": record.get("timestamp") or datetime.utcnow().isoformat(),
    }


@router.post("/analyse")
def analyse_review(payload: ReviewAnalysisRequest):
    analysis = analyze_review_text(payload.review_text)

    return {
        "review_text": payload.review_text,
        "sentiment": analysis["sentiment"],
        "confidence": analysis["confidence"],
        "source": analysis["source"],
    }

# -----------------------------
# POST - Submit Review
# -----------------------------
@router.post("/submit")
def submit_review(review: Review):
    reviews = read_reviews()

    inferred_sentiment = review.sentiment
    if not inferred_sentiment:
        analysis = analyze_review_text(review.comment)
        inferred_sentiment = analysis.get("sentiment")

    new_review = {
        "id": str(uuid.uuid4()),
        "username": review.username,
        "rating": max(1, min(5, review.rating)),
        "comment": review.comment,
        "sentiment": inferred_sentiment,
        "car_key": review.car_key,
        "timestamp": datetime.utcnow().isoformat()
    }

    reviews.append(new_review)
    write_reviews(reviews)

    return {
        "message": "Review submitted successfully",
        "review": new_review
    }


# -----------------------------
# GET - List Reviews (Paginated)
# -----------------------------
@router.get("/")
def get_reviews(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1),
    car_key: str | None = None
):
    reviews = [_normalize_review_record(r) for r in read_reviews() if isinstance(r, dict)]

    if car_key:
        reviews = [r for r in reviews if r.get("car_key") == car_key]

    sorted_reviews = sorted(
        reviews,
        key=lambda r: r.get("timestamp", "1970-01-01"),
        reverse=True
    )

    start = (page - 1) * limit
    end = start + limit
    paginated_reviews = sorted_reviews[start:end]

    return {
        "total_reviews": len(sorted_reviews),
        "page": page,
        "limit": limit,
        "data": paginated_reviews
    }


# -----------------------------
# DELETE - Remove Review
# -----------------------------
@router.delete("/{review_id}")
def delete_review(review_id: str):
    reviews = read_reviews()

    updated_reviews = [r for r in reviews if r.get("id") != review_id]

    if len(updated_reviews) == len(reviews):
        return {"message": "Review not found"}

    write_reviews(updated_reviews)

    return {"message": "Review deleted successfully"}