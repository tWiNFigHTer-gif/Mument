"""
Augments the training dataset with strong-negative EV reviews.
These cover clear negative language (very bad, poor, terrible, awful)
that the model started misclassifying as neutral after neutral augmentation.
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

script_dir = Path(__file__).parent
DATA_PATH = script_dir / "../../data/processed/swt/reviews_balanced_clean.json"
TARGET = 200  # Target per class after rebalancing

STRONG_NEGATIVES = [
    # "very bad" phrasing
    {"review_id": "aug_neg_001", "text": "It shows a very bad performance for long rides", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_002", "text": "Very bad battery life, dies before I reach my destination", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_003", "text": "The suspension is very bad on uneven roads", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_004", "text": "Very bad charging infrastructure support in my city", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_005", "text": "Customer service was very bad, no help at all", "rating": 1, "label": 0, "source": "augmented"},
    # "poor" phrasing
    {"review_id": "aug_neg_006", "text": "Poor range makes this EV impractical for anything beyond city use", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_007", "text": "Poor build quality for a car at this price point", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_008", "text": "Overall poor performance — I am very disappointed", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_009", "text": "Poor after-sales support from the dealership", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_010", "text": "Poor real-world range, far below what was advertised", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_011", "text": "Poor reliability — had to visit the service center three times this year", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_012", "text": "Poor acceleration compared to competitors in the same segment", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_013", "text": "Poor highway performance, range drops drastically at speed", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_014", "text": "Poor braking performance, longer stopping distance than expected", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_015", "text": "Poor software quality, constant glitches and crashes in the system", "rating": 1, "label": 0, "source": "augmented"},
    # "terrible" phrasing
    {"review_id": "aug_neg_016", "text": "Terrible experience with charging — the station was never available", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_017", "text": "Terrible on long trips, desperate need for more charging points", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_018", "text": "Terrible ride quality on highways, very uncomfortable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_019", "text": "Terrible customer experience — no one answers at the service center", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_020", "text": "Terrible value for money, I expected far more", "rating": 1, "label": 0, "source": "augmented"},
    # "awful" phrasing
    {"review_id": "aug_neg_021", "text": "Awful performance in cold weather, range drops by half", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_022", "text": "Awful interior quality for the price, plastics feel cheap", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_023", "text": "Awful driving experience on long highway trips", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_024", "text": "The app is awful, inaccurate battery readings all the time", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_025", "text": "Awful noise at high speeds, road noise is unbearable", "rating": 1, "label": 0, "source": "augmented"},
    # "horrible" phrasing
    {"review_id": "aug_neg_026", "text": "Horrible battery degradation after just one year of ownership", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_027", "text": "Horrible service center experience, waited weeks for a simple repair", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_028", "text": "Horrible charging speed at public stations in my area", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_029", "text": "Horrible performance in rain, stability control is unreliable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_030", "text": "Horrible resale value — lost a huge amount in just two years", "rating": 1, "label": 0, "source": "augmented"},
    # "disappointing" phrasing
    {"review_id": "aug_neg_031", "text": "Very disappointing range for a car marketed as long-distance capable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_032", "text": "Deeply disappointing experience overall — not what I expected", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_033", "text": "The performance is disappointing compared to what was promised", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_034", "text": "Quite disappointing for a flagship EV model in this price range", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_035", "text": "Hugely disappointing after all the hype around this EV", "rating": 1, "label": 0, "source": "augmented"},
    # "not good" / "not great" strong negative context
    {"review_id": "aug_neg_036", "text": "Performance on long rides is just not good enough for highway use", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_037", "text": "The range is not good for anyone who drives more than 150km a day", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_038", "text": "The EV is simply not good — reliability issues from day one", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_039", "text": "Not good at all — multiple faults in the first six months", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_040", "text": "Performance in summer heat is not good, battery overheats quickly", "rating": 1, "label": 0, "source": "augmented"},
    # Strong complaint phrasing
    {"review_id": "aug_neg_041", "text": "I regret buying this EV, it has been nothing but problems", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_042", "text": "Worst purchase I have made — the car constantly lets me down", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_043", "text": "A complete waste of money — unacceptable quality control", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_044", "text": "I am extremely frustrated with how unreliable this car has been", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_045", "text": "Completely let down by this EV — nothing as advertised", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_046", "text": "Absolutely terrible on long distance — range anxiety is real", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_047", "text": "I would not recommend this EV to anyone, too many issues", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_048", "text": "Severely overpriced for the poor quality on offer", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_049", "text": "The car broke down twice in the first year — unacceptable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_050", "text": "Massive disappointment — the range on this car is dangerously low", "rating": 1, "label": 0, "source": "augmented"},
    # Long ride / highway specific negatives
    {"review_id": "aug_neg_051", "text": "Very poor for long drives, charging stops every 150km kill the trip", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_052", "text": "Terrible on highways — battery drains fast and charging is slow", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_053", "text": "Completely impractical for long rides, defeats the purpose", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_054", "text": "Very bad on motorways — constant range anxiety is exhausting", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_055", "text": "Poor highway range, barely makes 180km at motorway speeds", "rating": 1, "label": 0, "source": "augmented"},
    # Reliability and breakdown phrasing
    {"review_id": "aug_neg_056", "text": "Very unreliable, had electrical failures multiple times", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_057", "text": "Deeply flawed vehicle — safety systems trigger randomly", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_058", "text": "The software crashes regularly, a serious safety concern", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_059", "text": "Unreliable charging port, often fails to connect", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_060", "text": "The brakes feel unsafe — too much play and inconsistent response", "rating": 1, "label": 0, "source": "augmented"},
    # Comfort negative
    {"review_id": "aug_neg_061", "text": "Very uncomfortable on long drives, back pain after an hour", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_062", "text": "Poor seat support on long journeys, deeply uncomfortable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_063", "text": "Terrible ride comfort on highways, the suspension ruins long trips", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_064", "text": "Very bad noise insulation — wind and road noise are intolerable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_065", "text": "Poor climate control, cabin gets unbearably hot in summer", "rating": 1, "label": 0, "source": "augmented"},
    # Cost and value negatives
    {"review_id": "aug_neg_066", "text": "Terrible value for money — far too expensive for what you get", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_067", "text": "Hugely overpriced for its poor feature set", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_068", "text": "Very poor cost-to-performance ratio for an EV in this segment", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_069", "text": "Not worth the asking price — very bad deal overall", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_070", "text": "Very high upfront cost for a poorly performing car", "rating": 1, "label": 0, "source": "augmented"},
    # Charging infrastructure negative
    {"review_id": "aug_neg_071", "text": "Very bad public charging coverage — always out of order", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_072", "text": "Terrible charging network, waited two hours at broken chargers", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_073", "text": "Poor fast-charging capability, takes too long for daily convenience", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_074", "text": "Awful charging experience — connectors fail constantly", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_075", "text": "Very bad charging cable quality — already fraying after a month", "rating": 1, "label": 0, "source": "augmented"},
    # Generic strong negative summary
    {"review_id": "aug_neg_076", "text": "A very poor EV overall, do not buy", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_077", "text": "Very poor experience from purchase to ownership", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_078", "text": "Terrible car — looks good on paper, fails in real use", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_079", "text": "Awful car, nothing works as advertised", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_080", "text": "A horrible experience from day one with this EV", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_081", "text": "Absolutely not worth it — very bad long term reliability", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_082", "text": "Very bad overall — multiple issues, poor support, low range", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_083", "text": "Poor all around — performance, range, service, everything", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_084", "text": "Terrible engineering decisions made this car frustrating to own", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_085", "text": "This EV has been a nightmare — poor quality and poor support", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_086", "text": "I deeply regret this purchase — very bad car for the money", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_087", "text": "Very bad performance compared to what competitors offer", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_088", "text": "Shocking how bad the range is in real-world conditions", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_089", "text": "Disappointingly bad in almost every area that matters", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_090", "text": "A very bad car — I would not buy it again under any circumstances", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_091", "text": "The performance is so poor it is actually dangerous on highways", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_092", "text": "Incredibly poor performance for a modern EV — unacceptable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_093", "text": "Very bad experience — the car never lived up to expectations", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_094", "text": "Dreadful reliability — constant breakdowns ruin the ownership experience", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_095", "text": "The worst EV in its price range — poor value and poor performance", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_096", "text": "Very bad build quality for an electric vehicle at this price", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_097", "text": "Extremely poor ownership experience — not what I signed up for", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_098", "text": "Very disappointing performance on long drives, completely unsuitable", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_099", "text": "Awful car for the money — stay away from this EV", "rating": 1, "label": 0, "source": "augmented"},
    {"review_id": "aug_neg_100", "text": "Very bad car — poor range, poor quality, poor support, avoid", "rating": 1, "label": 0, "source": "augmented"},
]


def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    before = Counter(r["label"] for r in data)
    print(f"Before: neg={before[0]}, neu={before[1]}, pos={before[2]}")

    # Add strong negatives
    data.extend(STRONG_NEGATIVES)

    # Separate and rebalance to TARGET
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
    print("Now run: python -m ai.supervised_sentiment.train")

if __name__ == "__main__":
    main()
