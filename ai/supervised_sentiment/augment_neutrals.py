"""
Adds hedged-language neutral training examples to reviews_balanced_clean.json.
These are the kinds of reviews DistilBERT currently misclassifies as negative
because they contain phrases like "not too great", "okay", "could be better".
"""

import json
from pathlib import Path

script_dir = Path(__file__).parent
DATA_PATH = script_dir / "../../data/processed/swt/reviews_balanced_clean.json"

# Hedged neutral reviews the model struggles with
HEDGED_NEUTRALS = [
    {"review_id": "aug_n_001", "text": "The car is okay, not too bad, not too great, is okay for city use", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_002", "text": "Not bad, not great — it gets the job done for daily commuting", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_003", "text": "It's okay for the price, nothing to rave about but not disappointing either", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_004", "text": "Could be better, could be worse — an average EV overall", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_005", "text": "Not the best EV out there but definitely not the worst", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_006", "text": "Decent enough for everyday use, though nothing exceptional", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_007", "text": "The car is alright, I have no major complaints but nothing wowed me", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_008", "text": "It's fine for city driving, not ideal for long trips", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_009", "text": "Range is okay, not spectacular, but acceptable for my use case", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_010", "text": "Mixed feelings — some things are great, others are disappointing", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_011", "text": "I guess it is what it is, average performance expected at this price", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_012", "text": "Not exactly thrilling to drive but reliable enough", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_013", "text": "The experience is so-so — not bad enough to regret the purchase", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_014", "text": "Charging speed is okay, not fast not slow, acceptable for home use", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_015", "text": "Neither impressed nor disappointed, just an average overall package", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_016", "text": "It works fine, but I expected a bit more for the money", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_017", "text": "The car handles okay, not the smoothest but far from rough", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_018", "text": "Battery life is decent enough, not great, but I manage daily", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_019", "text": "The interior is passable, not premium looking but not terrible either", "rating": 3, "label": 1, "source": "augmented"},
    {"review_id": "aug_n_020", "text": "Overall an acceptable car, wouldn't say I love it or hate it", "rating": 3, "label": 1, "source": "augmented"},
]

def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    before = len(data)
    neutral_before = sum(1 for r in data if r["label"] == 1)

    data.extend(HEDGED_NEUTRALS)

    neutral_after = sum(1 for r in data if r["label"] == 1)
    print(f"Neutral examples: {neutral_before} → {neutral_after} (+{len(HEDGED_NEUTRALS)})")
    print(f"Total dataset: {before} → {len(data)}")

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved to {DATA_PATH}")
    print("Now run: python -m ai.supervised_sentiment.train")

if __name__ == "__main__":
    main()
