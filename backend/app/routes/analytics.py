# Analytics routes
import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["analytics"])

BASE_DIR = Path(__file__).resolve().parents[3]
SUMMARY_PATH = BASE_DIR / "data" / "artifacts" / "chatbot_summary.json"


@router.get("/summary")
def get_summary():

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    return {
        "total_reviews": summary["total_reviews"],
        "positive": summary["sentiment_distribution"].get("positive", 0),
        "negative": summary["sentiment_distribution"].get("negative", 0),
        "neutral": summary["sentiment_distribution"].get("neutral", 0),
    }

