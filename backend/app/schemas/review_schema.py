from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Review(BaseModel):
    username: str
    rating: int
    comment: str
    sentiment: Optional[str] = None

class ReviewResponse(BaseModel):
    id: str
    username: str
    rating: int
    comment: str
    sentiment: Optional[str]
    timestamp: datetime