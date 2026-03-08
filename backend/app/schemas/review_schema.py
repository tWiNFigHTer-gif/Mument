from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Review(BaseModel):
    username: str
    rating: int
    comment: str
    sentiment: Optional[str] = None


class ReviewAnalysisRequest(BaseModel):
    review_text: str


class ReviewAnalysisResponse(BaseModel):
    review_text: str
    sentiment: str
    confidence: float
    source: str

class ReviewResponse(BaseModel):
    id: str
    username: str
    rating: int
    comment: str
    sentiment: Optional[str]
    timestamp: datetime