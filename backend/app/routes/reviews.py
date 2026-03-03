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
