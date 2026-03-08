# Roadmap generator
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

SUMMARY_PATH = BASE_DIR / "data" / "artifacts" / "chatbot_summary.json"
CLUSTER_META_PATH = BASE_DIR / "data" / "artifacts" / "cluster_metadata.json"
OUTPUT_PATH = BASE_DIR / "data" / "artifacts" / "roadmap_output.json"

CLUSTER_LABELS = {
    "0": "Interior Quality",
    "1": "Battery Range",
    "2": "EV Comfort & Feel",
    "3": "General Experience",
    "4": "Charging & Usability"
}

def generate_roadmap():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    with open(CLUSTER_META_PATH) as f:
        cluster_meta = json.load(f)

    worst_id = str(summary["worst_cluster_id"])
    worst_keywords = summary["worst_cluster_keywords"]
    worst_label = CLUSTER_LABELS.get(worst_id, f"Cluster {worst_id}")
    negative_pct = round(summary["negative_ratio"] * 100, 1)

    roadmap = {
        "root_problem_area": (
            f"The most critical area is '{worst_label}'. "
            f"{negative_pct}% of reviews are negative, primarily around: "
            f"{', '.join(worst_keywords[:5])}."
        ),
        "immediate_actions_0_3_months": [
            f"Conduct user research sessions focused on '{worst_label}' complaints",
            f"Review top negative feedback keywords: {', '.join(worst_keywords[:3])}",
            "Prioritise fixes for the worst-rated cluster in the next sprint",
            "Set up automated sentiment monitoring on incoming reviews"
        ],
        "short_term_actions_3_6_months": [
            f"Launch targeted improvement initiative for '{worst_label}'",
            "A/B test product changes against baseline satisfaction scores",
            "Increase positive review rate by 10% through addressed pain points"
        ],
        "long_term_actions_6_12_months": [
            "Build continuous feedback loop between support and product teams",
            "Integrate NLP sentiment pipeline into product review dashboard",
            "Quarterly re-clustering to track evolving customer concerns"
        ],
        "cluster_summary": {
            cid: {
                "label": CLUSTER_LABELS.get(cid, f"Cluster {cid}"),
                "keywords": cluster_meta[cid]
            }
            for cid in cluster_meta
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(roadmap, f, indent=2)

    print("Roadmap generated successfully.")
    return roadmap

if __name__ == "__main__":
    generate_roadmap()