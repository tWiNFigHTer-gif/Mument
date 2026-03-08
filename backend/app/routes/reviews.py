# Reviews routes

from fastapi import APIRouter
from pydantic import BaseModel
from ai.pipeline import analyse

router = APIRouter()

class ReviewInput(BaseModel):
    text: str

@router.post("/reviews/analyse")
def analyse_review(review: ReviewInput):
    """
    Accepts review text
    sentiment label, decided_by return chyym

    """
    result = analyse(review.text)
    return result
from fastapi import APIRouter, Query
from app.schemas.review_schema import Review
from app.services.json_storage import read_reviews, write_reviews
import uuid
from datetime import datetime

router = APIRouter()


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

    return {"message": "Review submitted successfully"}


# -----------------------------
# GET - List Reviews (Paginated)
# -----------------------------
@router.get("/")
def get_all_reviews(
    page: int = Query(1, ge=1),
    limit: int = Query(5, ge=1)
):
    reviews = read_reviews()

    # Sort newest first
    sorted_reviews = sorted(
        reviews,
        key=lambda r: r["timestamp"],
        reverse=True
    )

    # Pagination logic
    start = (page - 1) * limit
    end = start + limit

    paginated_data = sorted_reviews[start:end]

    return {
        "total": len(reviews),
        "page": page,
        "limit": limit,
        "data": paginated_data
    }


# -----------------------------
# DELETE - Remove Review
# -----------------------------
@router.delete("/{review_id}")
def delete_review(review_id: str):
    reviews = read_reviews()

    updated_reviews = [r for r in reviews if r["id"] != review_id]

    if len(updated_reviews) == len(reviews):
        return {"message": "Review not found"}

    write_reviews(updated_reviews)

    return {"message": "Review deleted successfully"}
