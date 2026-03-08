"""
Adds intensifier-phrase training examples (so good, very disappointing, etc.)
that the model is currently misclassifying as neutral.
Keeps classes balanced at 200 each after adding.
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

script_dir = Path(__file__).parent
DATA_PATH = script_dir / "../../data/processed/swt/reviews_balanced_clean.json"
TARGET = 200

# Strong POSITIVE with intensifiers
INTENSIFIER_POSITIVES = [
    {"review_id": "int_p_001", "text": "So good — the EV exceeded every expectation I had", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_002", "text": "So good for city driving, I absolutely love this car", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_003", "text": "This EV is so good, best purchase I have ever made", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_004", "text": "The range is so good, I rarely need to charge more than once a week", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_005", "text": "So good in every way — smooth, quiet, powerful and efficient", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_006", "text": "The performance is so good it makes other cars feel outdated", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_007", "text": "So incredibly good — I recommend this EV to everyone I know", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_008", "text": "This car is so good, I regret not buying it sooner", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_009", "text": "The software is so good, over-the-air updates keep improving everything", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_010", "text": "Charging speed is so good, 80 percent in under 30 minutes", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_011", "text": "Really good build quality, feels very premium inside and out", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_012", "text": "Very good range for daily use — no range anxiety at all", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_013", "text": "Extremely good performance, the instant torque is addictive", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_014", "text": "Very good car overall — would buy again without hesitation", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_015", "text": "Really good experience from purchase to everyday driving", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_016", "text": "The cabin is really good, premium materials and very quiet", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_017", "text": "Very good handling both in city traffic and on expressways", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_018", "text": "Extremely good value for money — far exceeded my expectations", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_019", "text": "Very good ownership experience — smooth, reliable, joyful", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_020", "text": "Really good EV, the drive quality is exceptionally smooth", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_021", "text": "Incredibly good service, delivered on time with full explanation", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_022", "text": "The EV is so impressive, way better than my old petrol car", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_023", "text": "Super good comfort levels even on long highway drives", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_024", "text": "The regenerative braking is really good, saves a ton of energy", "rating": 5, "label": 2, "source": "augmented"},
    {"review_id": "int_p_025", "text": "Very good battery life — three years in and barely any degradation", "rating": 5, "label": 2, "source": "augmented"},
]

# Strong NEGATIVE with intensifiers
INTENSIFIER_NEGATIVES = [
    {"review_id": "int_n_001", "text": "Very disappointing — expected much better from this brand", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_002", "text": "So disappointing, the car does not match what was advertised at all", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_003", "text": "Very disappointing range — barely 150km in real-world conditions", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_004", "text": "Hugely disappointing charging speed, takes hours for a full charge", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_005", "text": "Very disappointing service experience, waited three weeks for parts", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_006", "text": "So disappointing overall — not worth the premium price at all", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_007", "text": "Very disappointing performance at highway speeds, lots of noise", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_008", "text": "Extremely disappointing — the features promised were not delivered", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_009", "text": "Very disappointing build quality — panels misaligned on delivery", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_010", "text": "So disappointing in cold weather, range drops by 40 percent", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_011", "text": "Really bad ownership experience, multiple visits to service center", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_012", "text": "Very bad decision to buy this — the reliability is shocking", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_013", "text": "Really bad cabin quality, the materials scratch and fade quickly", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_014", "text": "So bad on long rides, the suspension is stiff and uncomfortable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_015", "text": "Very bad software — crashes, freezes and requires constant reboots", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_016", "text": "Really disappointing interior — looks cheap and feels cheap", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_017", "text": "Truly disappointing — nothing about this car justifies the cost", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_018", "text": "Very bad acceleration at highway speed, dangerous to overtake", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_019", "text": "So frustrating to own — constant bugs and poor support", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_020", "text": "Really frustrating charging experience — stations always busy or broken", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_021", "text": "Very frustrating reliability — breakdowns during important trips", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_022", "text": "Deeply unhappy with this car — poor quality and poor support", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_023", "text": "Very unhappy with my purchase — feel misled by the marketing", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_024", "text": "So unhappy — this EV has caused nothing but stress", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "int_n_025", "text": "Very bad experience overall — would warn everyone away from this car", "rating": 1, "label": 0, "source": "augmented"},
]


def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    before = Counter(r["label"] for r in data)
    print(f"Before: neg={before[0]}, neu={before[1]}, pos={before[2]}")

    data.extend(INTENSIFIER_POSITIVES)
    data.extend(INTENSIFIER_NEGATIVES)

    negatives = [r for r in data if r["label"] == 0]
    neutrals  = [r for r in data if r["label"] == 1]
    positives = [r for r in data if r["label"] == 2]

    if len(negatives) > TARGET:
        negatives = random.sample(negatives, TARGET)
    if len(neutrals) > TARGET:
        neutrals = random.sample(neutrals, TARGET)
    if len(positives) > TARGET:
        positives = random.sample(positives, TARGET)

    balanced = negatives + neutrals + positives
    random.shuffle(balanced)

    after = Counter(r["label"] for r in balanced)
    print(f"After:  neg={after[0]}, neu={after[1]}, pos={after[2]}")
    print(f"Total: {len(balanced)} reviews")

    with open(DATA_PATH, "w") as f:
        json.dump(balanced, f, indent=2)

    print(f"✅ Saved to {DATA_PATH}")

if __name__ == "__main__":
    main()
