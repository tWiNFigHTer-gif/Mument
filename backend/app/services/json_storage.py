import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "../data/reviews.json")

def read_reviews():
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "w") as file:
            json.dump([], file)

    with open(FILE_PATH, "r") as file:
        return json.load(file)

def save_review(review):
    reviews = read_reviews()
    reviews.append(review)

    with open(FILE_PATH, "w") as file:
        json.dump(reviews, file, indent=4)