# Chatbot rules
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

SUMMARY_PATH = BASE_DIR / "data" / "artifacts" / "chatbot_summary.json"
ROADMAP_PATH = BASE_DIR / "data" / "artifacts" / "roadmap_output.json"


def generate_response(user_query: str):

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    with open(ROADMAP_PATH) as f:
        roadmap = json.load(f)

    query = user_query.lower()

    if "problem" in query or "issue" in query:
        reply = f"Customers are mainly facing issues related to {summary['worst_cluster_keywords'][0]}."

    elif "improve" in query or "fix" in query:
        reply = roadmap["root_problem_area"]

    else:
        reply = "Based on customer reviews, here is a recommended improvement strategy."

    return {
        "reply": reply,
        "roadmap": roadmap["immediate_actions_0_3_months"]
    }
