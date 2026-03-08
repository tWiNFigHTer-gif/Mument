from fastapi import APIRouter, Query
from app.schemas.review_schema import Review, ReviewAnalysisRequest
from app.services.json_storage import read_reviews, write_reviews
from app.services.sentiment_service import analyze_review_text
import uuid
from datetime import datetime

router = APIRouter()


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

    new_review = {
        "id": str(uuid.uuid4()),
        "username": review.username,
        "rating": review.rating,
        "comment": review.comment,
        "sentiment": review.sentiment,
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
    limit: int = Query(5, ge=1)
):
    reviews = read_reviews()

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