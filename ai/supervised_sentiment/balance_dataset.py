"""
Balance the training dataset to ~200 examples per class.
- Neutral (1): currently 40 → add 160 synthetic examples → 200
- Negative (0): currently 242 → downsample to 200
- Positive (2): currently 257 → downsample to 200
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

script_dir = Path(__file__).parent
DATA_PATH = script_dir / "../../data/processed/swt/reviews_balanced_clean.json"
TARGET = 200

# 160 diverse synthetic neutral EV reviews
SYNTHETIC_NEUTRALS = [
    # Hedged / mixed opinion
    {"review_id": "syn_n_001", "text": "The EV is neither amazing nor terrible, just average", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_002", "text": "Some days I like it, some days I don't — an average car", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_003", "text": "It has good points and bad points, overall a balanced vehicle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_004", "text": "Not bad for an EV but not the best I have driven either", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_005", "text": "Meets expectations, nothing more nothing less", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_006", "text": "The car is okay for what I paid, average in every way", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_007", "text": "Would not say I love it or hate it — it is a fine car", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_008", "text": "Pros and cons balance each other out, hard to recommend or discourage", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_009", "text": "The EV does its job without any surprises, positive or negative", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_010", "text": "Average build quality, average performance, average experience", "rating": 3, "label": 1, "source": "synthetic"},
    # Range / charging neutral
    {"review_id": "syn_n_011", "text": "Range is acceptable for city driving, not ideal for highways", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_012", "text": "Charging speed is mid — fast enough but not impressive", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_013", "text": "Battery performance is what I expected for this class of vehicle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_014", "text": "The range on paper versus real world is close enough, no major surprise", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_015", "text": "Home charging works fine, public chargers are hit or miss", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_016", "text": "I get about 200km range which is okay for my daily needs", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_017", "text": "Charging at night is convenient, during the day a bit of a hassle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_018", "text": "Battery degrades at an average rate, nothing alarming", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_019", "text": "The range is fine for weekdays, tight on weekend road trips", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_020", "text": "Fast charging works reliably, slow charging is frustratingly slow", "rating": 3, "label": 1, "source": "synthetic"},
    # Comfort / interior neutral
    {"review_id": "syn_n_021", "text": "The interior is functional but does not feel premium", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_022", "text": "Seats are comfortable for short drives, less so for long ones", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_023", "text": "Cabin noise at highway speeds is acceptable but noticeable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_024", "text": "Infotainment works as expected, nothing groundbreaking", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_025", "text": "Rear legroom is adequate but not generous", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_026", "text": "Boot space is average, not great for family trips", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_027", "text": "The display is clear but the UI takes time to get used to", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_028", "text": "Climate control works well in mild weather, struggles in extreme heat", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_029", "text": "Ride quality on smooth roads is good, bumpy on rough surfaces", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_030", "text": "The steering is light and easy, some may prefer more feedback", "rating": 3, "label": 1, "source": "synthetic"},
    # Performance neutral
    {"review_id": "syn_n_031", "text": "Acceleration is adequate, faster than petrol, slower than premium EVs", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_032", "text": "The drive is smooth but unremarkable — predictable everyday transport", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_033", "text": "Performance is middle of the road, not sporty but not sluggish either", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_034", "text": "Regenerative braking is okay once you adjust to it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_035", "text": "Handles city roads well but feels average on highway bends", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_036", "text": "Braking distance is normal, nothing impressive or worrying", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_037", "text": "Ground clearance is sufficient for most city roads in my area", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_038", "text": "The EV performs as expected for a mid-range electric vehicle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_039", "text": "Power delivery feels fine but not exciting for a car this price", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_040", "text": "The car does what it says on the box — no more, no less", "rating": 3, "label": 1, "source": "synthetic"},
    # Service / ownership neutral
    {"review_id": "syn_n_041", "text": "Service experience was average, no major issues but nothing stood out", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_042", "text": "Ownership cost is moderate, not as cheap as expected", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_043", "text": "Dealer support is okay, not exceptional", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_044", "text": "Software updates are helpful but some introduce new issues", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_045", "text": "The app connectivity works most of the time, occasional glitches", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_046", "text": "Maintenance is simpler than a petrol car but not zero-cost", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_047", "text": "Warranty coverage is standard — nothing exceptional in the fine print", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_048", "text": "The resale value is uncertain, as with most new-era EVs", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_049", "text": "Running costs savings are real but modest in my use case", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_050", "text": "The overall ownership experience is satisfactory, not remarkable", "rating": 3, "label": 1, "source": "synthetic"},
    # More okay/fine phrasing
    {"review_id": "syn_n_051", "text": "The car is okay I guess, serves its purpose without frills", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_052", "text": "It is what it is — a practical EV with no standout qualities", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_053", "text": "Good enough for daily needs, borderline for anything else", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_054", "text": "I am neither happy nor unhappy with the purchase", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_055", "text": "The experience has been pretty standard so far", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_056", "text": "Feels like a competent but uninspiring choice in the EV market", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_057", "text": "Not too impressed but also not disappointed — middle of the road", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_058", "text": "Has its pros and cons — on balance it is acceptable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_059", "text": "I would describe my experience as neutral — nothing stood out", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_060", "text": "The car is tolerable for what I paid, not great not horrible", "rating": 3, "label": 1, "source": "synthetic"},
    # Slightly positive lean but still neutral
    {"review_id": "syn_n_061", "text": "Decent enough car for the segment, would consider it again", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_062", "text": "A reasonable choice if you are not looking for anything special", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_063", "text": "Does what I need without complaint — that is about it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_064", "text": "Not the worst EV on the market, not the best — squarely in the middle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_065", "text": "The technology is current but not cutting edge", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_066", "text": "A sensible choice for practical buyers not chasing excitement", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_067", "text": "The build quality feels average — solid but not impressive", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_068", "text": "Suitable for buyers who want an EV without any fuss", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_069", "text": "Gets the job done efficiently, not thrilling but reliable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_070", "text": "The car is safe, practical, ordinary — completely average", "rating": 3, "label": 1, "source": "synthetic"},
    # Slightly negative lean but still neutral
    {"review_id": "syn_n_071", "text": "Some features feel half-baked but core driving is acceptable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_072", "text": "Expected more innovation for this price but it does the job", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_073", "text": "A few quirks that are annoying but not dealbreakers", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_074", "text": "The ride is a little stiff but tolerable for city use", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_075", "text": "Charging network could be better but I manage with what is available", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_076", "text": "The UI has a learning curve but works fine once you get used to it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_077", "text": "Some cost-cutting is visible in the interior but not a dealbreaker", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_078", "text": "Not everything is perfect but nothing is seriously wrong either", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_079", "text": "The car feels a bit underwhelming but there is nothing broken", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_080", "text": "Average at everything — that can be a good thing depending on your needs", "rating": 3, "label": 1, "source": "synthetic"},
    # Everyday use / practical neutral
    {"review_id": "syn_n_081", "text": "Works well as a second car, less practical as a primary vehicle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_082", "text": "Good for urban commuting, less suited for long-distance travel", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_083", "text": "The EV is dependable for predictable daily use", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_084", "text": "Perfectly adequate for my 25km daily commute, nothing special", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_085", "text": "An okay commuter car with no remarkable strengths or weaknesses", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_086", "text": "My experience over 6 months has been uneventful — which is neutral", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_087", "text": "The car does not excite me but I have no plans to switch", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_088", "text": "It covers my commute, that is really all I can say", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_089", "text": "Reliable transportation, no love affair with the vehicle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_090", "text": "Functional daily driver, does not inspire passion or frustration", "rating": 3, "label": 1, "source": "synthetic"},
    # Comparative neutral
    {"review_id": "syn_n_091", "text": "Similar to other EVs in this price range, nothing sets it apart", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_092", "text": "On par with competitors — not better, not worse", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_093", "text": "About what you would expect from a mass-market EV", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_094", "text": "Industry-standard features, no surprises on either end", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_095", "text": "Comparable to my previous petrol car in terms of overall satisfaction", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_096", "text": "Falls in line with what you expect from a budget EV", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_097", "text": "Matches the segment average — nothing differentiates it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_098", "text": "Competes reasonably with rivals but does not lead the pack", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_099", "text": "Average among its peers, which is not necessarily a bad thing", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_100", "text": "Not the standout choice in its category but certainly not the worst", "rating": 3, "label": 1, "source": "synthetic"},
    # More varied phrasing
    {"review_id": "syn_n_101", "text": "I have mixed feelings — the drive is nice but the range lets me down", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_102", "text": "The positives and negatives roughly cancel out for me", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_103", "text": "Good ideas, average execution — that sums up this EV", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_104", "text": "I tolerate it rather than enjoy it — but no reason to return it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_105", "text": "Three stars out of five — genuinely in the middle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_106", "text": "Would not go out of my way to recommend or warn others away", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_107", "text": "The car is fine — I have nothing strong to say about it", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_108", "text": "Satisfactory in most areas, slightly disappointing in a few", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_109", "text": "The good outweighs the bad only marginally — call it neutral", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_110", "text": "My overall impression is a flat three out of five", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_111", "text": "Not every day is great with this car, not every day is bad", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_112", "text": "The EV is workable — a middle-of-the-road choice", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_113", "text": "It serves a purpose without leaving a strong impression", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_114", "text": "Moderate in every way — range, comfort, performance", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_115", "text": "The ownership experience has been uneventful so far", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_116", "text": "The car is reasonable — not a regret, not a highlight", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_117", "text": "An honest middling review — the car earns a solid meh", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_118", "text": "I bought this expecting average and average is exactly what I got", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_119", "text": "The EV performs satisfactorily — no cheers, no complaints", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_120", "text": "My feelings toward this car can be summarised as indifferent", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_121", "text": "Adequate vehicle for moderate usage patterns", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_122", "text": "Unremarkable but dependable daily transport", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_123", "text": "The car sits comfortably in the middle tier of EVs", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_124", "text": "Neither a strong yes nor a strong no — just an okay car", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_125", "text": "It satisfies basic requirements without exceeding them", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_126", "text": "An acceptable EV for someone who does not need to be wowed", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_127", "text": "The EV leaves me feeling nothing strong — neutral experience", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_128", "text": "Reliable and ordinary — exactly what some buyers need", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_129", "text": "I have driven better and I have driven worse than this EV", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_130", "text": "The car does not push any boundaries, pleasant but forgettable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_131", "text": "Features are what I expected for a mid-tier EV — fine", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_132", "text": "Nothing jumps out as exceptional but nothing is a dealbreaker", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_133", "text": "The EV checked the boxes I needed, nothing extra was on offer", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_134", "text": "The driving experience is consistent and predictable — average", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_135", "text": "Overall the car is just okay and that is fine by me", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_136", "text": "A straightforward no-frills EV that does its job", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_137", "text": "Would rate it three out of five — true to the middle", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_138", "text": "My experience was ordinary — I expected more excitement", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_139", "text": "The EV is passable, I have adapted to its limitations", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_140", "text": "Not standout, not terrible — safely in the neutral zone", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_141", "text": "The car is alright for what it is, a practical commuter", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_142", "text": "A mediocre EV in the best sense of the word — consistently average", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_143", "text": "The car is fine for city use, not impressive beyond that", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_144", "text": "Handles routine driving well, struggles to impress otherwise", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_145", "text": "The overall package is fair value for money — nothing special", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_146", "text": "Comfortably average electric car with no surprises", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_147", "text": "I give it a solid middle score — balanced strengths and weaknesses", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_148", "text": "The car is solid without being exciting — a neutral ownership", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_149", "text": "Ordinary in all the right ways for a daily commuter EV", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_150", "text": "A car that adequately fills its role without distinction", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_151", "text": "It is a satisfactory product — I have no intense opinion either way", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_152", "text": "The EV is reasonable on all fronts — a safe, boring choice", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_153", "text": "The car has not made me happy or sad in twelve months of ownership", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_154", "text": "Exactly as advertised — average specs delivering average results", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_155", "text": "The EV is okay for now — time will tell if I feel stronger", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_156", "text": "Acceptable range, acceptable comfort, acceptable price — just acceptable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_157", "text": "A car I would describe as tolerable rather than lovable", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_158", "text": "It has grown on me a little but I still consider it average", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_159", "text": "The EV is not going to blow you away but it will not let you down", "rating": 3, "label": 1, "source": "synthetic"},
    {"review_id": "syn_n_160", "text": "Overall a neutral ownership experience — average across the board", "rating": 3, "label": 1, "source": "synthetic"},
]


def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # Separate by class
    negatives = [r for r in data if r["label"] == 0]
    neutrals  = [r for r in data if r["label"] == 1]
    positives = [r for r in data if r["label"] == 2]

    print(f"Before: neg={len(negatives)}, neu={len(neutrals)}, pos={len(positives)}")

    # Add synthetic neutrals
    neutrals.extend(SYNTHETIC_NEUTRALS)

    # Downsample larger classes to TARGET
    if len(negatives) > TARGET:
        negatives = random.sample(negatives, TARGET)
    if len(positives) > TARGET:
        positives = random.sample(positives, TARGET)
    if len(neutrals) > TARGET:
        neutrals = random.sample(neutrals, TARGET)

    balanced = negatives + neutrals + positives
    random.shuffle(balanced)

    label_counts = Counter(r["label"] for r in balanced)
    print(f"After:  neg={label_counts[0]}, neu={label_counts[1]}, pos={label_counts[2]}")
    print(f"Total: {len(balanced)} reviews")

    with open(DATA_PATH, "w") as f:
        json.dump(balanced, f, indent=2)

    print(f"✅ Saved balanced dataset to {DATA_PATH}")
    print("Now run: python -m ai.supervised_sentiment.train")

if __name__ == "__main__":
    main()
