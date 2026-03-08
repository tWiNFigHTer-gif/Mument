"""
Data cleaning script for supervised_sentiment training data.
Removes off-topic reviews (non-EV/car related) from negative class
and deduplicates neutral examples to improve neutral detection.
"""

import json
from pathlib import Path
from collections import Counter

script_dir = Path(__file__).parent
INPUT_PATH = script_dir / "../../data/processed/swt/reviews_balanced.json"
OUTPUT_PATH = script_dir / "../../data/processed/swt/reviews_balanced_clean.json"

# Keywords that indicate the review is actually about cars/EVs
EV_KEYWORDS = [
    "car", "ev", "electric", "vehicle", "driving", "drive", "battery",
    "charging", "charge", "range", "motor", "acceleration", "brake",
    "seat", "cabin", "interior", "infotainment", "suspension", "wheel",
    "tire", "tyre", "torque", "speed", "mileage", "fuel", "engine",
    "autopilot", "tesla", "fsd", "service", "warranty", "dealer",
    "delivery", "maintenance", "comfort", "noise", "climate", "software",
    "update", "camera", "sensor", "safety", "performance", "cost",
    "price", "upfront", "affordable", "petrol", "eco", "green",
    "regenerative", "regen", "supercharger", "fast charg", "home charg",
    "road", "highway", "city", "trip", "journey", "ownership", "buy",
    "purchase", "model", "variant", "feature", "display", "screen"
]

def is_ev_related(text: str) -> bool:
    """Check if a review is about EVs/cars using keyword matching."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EV_KEYWORDS)

def clean_dataset(data: list) -> list:
    cleaned = []
    removed = []
    seen_neutral_texts = set()

    for review in data:
        text = review.get("text", "") or ""
        label = review.get("label")

        # For negatives: filter out off-topic reviews
        if label == 0:
            if is_ev_related(text):
                cleaned.append(review)
            else:
                removed.append(review)

        # For neutrals: deduplicate (keep first occurrence only)
        elif label == 1:
            norm_text = text.strip().lower()
            if norm_text not in seen_neutral_texts:
                seen_neutral_texts.add(norm_text)
                cleaned.append(review)
            else:
                removed.append(review)

        # Positives: keep all
        else:
            cleaned.append(review)

    return cleaned, removed

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Original dataset: {len(data)} reviews")
    label_counts = Counter(r["label"] for r in data)
    print(f"Before: {dict(sorted(label_counts.items()))}")

    cleaned, removed = clean_dataset(data)

    print(f"\nRemoved {len(removed)} reviews:")
    for r in removed[:10]:  # Show first 10 removed
        print(f"  [label={r['label']}] {r['text'][:80]}")
    if len(removed) > 10:
        print(f"  ... and {len(removed) - 10} more")

    label_counts_after = Counter(r["label"] for r in cleaned)
    print(f"\nAfter: {dict(sorted(label_counts_after.items()))}")
    print(f"Clean dataset: {len(cleaned)} reviews")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"\n✅ Saved cleaned data to: {OUTPUT_PATH}")
    print("Now run: python -m ai.supervised_sentiment.train_clean")

if __name__ == "__main__":
    main()
