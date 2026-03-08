# Analytics routes
from fastapi import APIRouter
from app.services.analytics_service import get_review_statistics

router = APIRouter()

@router.get("/stats")
def review_stats():
    return get_review_statistics()


@router.get("/summary")
def review_summary():
    stats = get_review_statistics()
    return {
        "total_reviews": stats["total_reviews"],
        "positive": stats["positive"],
        "negative": stats["negative"],
        "neutral": stats["neutral"],
    }