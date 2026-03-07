# Analytics routes
from fastapi import APIRouter
from app.services.analytics_service import get_review_statistics

router = APIRouter()

@router.get("/stats")
def review_stats():
    return get_review_statistics()