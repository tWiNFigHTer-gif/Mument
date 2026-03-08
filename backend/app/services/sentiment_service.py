from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


POSITIVE_KEYWORDS = {
    "amazing",
    "awesome",
    "best",
    "excellent",
    "fantastic",
    "good",
    "great",
    "love",
    "loved",
    "premium",
    "smooth",
}

NEGATIVE_KEYWORDS = {
    "awful",
    "bad",
    "boring",
    "costly",
    "disappointing",
    "expensive",
    "hate",
    "high",
    "poor",
    "problem",
    "terrible",
    "worst",
}


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _heuristic_sentiment(text: str) -> dict:
    lowered_text = text.lower()
    tokens = set(lowered_text.replace(",", " ").replace(".", " ").split())

    positive_hits = sum(1 for word in POSITIVE_KEYWORDS if word in tokens)
    negative_hits = sum(1 for word in NEGATIVE_KEYWORDS if word in tokens)

    if positive_hits > negative_hits:
        sentiment = "positive"
        confidence = 0.65
    elif negative_hits > positive_hits:
        sentiment = "negative"
        confidence = 0.65
    else:
        sentiment = "neutral"
        confidence = 0.5

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "source": "heuristic",
    }


@lru_cache(maxsize=1)
def _load_model_runtime():
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from ai.supervised_sentiment.infer import load_sentiment_model  # noqa: WPS433

    tokenizer, model = load_sentiment_model()
    return tokenizer, model


def analyze_review_text(text: str) -> dict:
    clean_text = text.strip()
    if not clean_text:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "source": "empty-input",
        }

    try:
        tokenizer, model = _load_model_runtime()

        from ai.supervised_sentiment.infer import predict_sentiment  # noqa: WPS433

        result = predict_sentiment(clean_text, tokenizer, model)
        return {
            "sentiment": _normalize_label(result["label"]),
            "confidence": round(float(result["score"]), 4),
            "source": "ml-model",
        }
    except Exception:
        return _heuristic_sentiment(clean_text)