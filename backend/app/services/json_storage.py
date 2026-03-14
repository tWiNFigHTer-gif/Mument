import json
import os

# Get path to reviews.json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "data", "reviews.json")


def read_reviews():
    """Read all reviews from JSON file"""
    if not os.path.exists(FILE_PATH):
        return []

    with open(FILE_PATH, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return []

    return data if isinstance(data, list) else []


def write_reviews(data):
    """Write updated reviews list to JSON file"""
    with open(FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)