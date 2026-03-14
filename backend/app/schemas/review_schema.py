from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Review(BaseModel):
    username: str
    rating: int
    comment: str
    sentiment: Optional[str] = None
    car_key: str


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
    car_key: str
    timestamp: datetime