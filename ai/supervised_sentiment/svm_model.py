import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter

# ---------- Base Path ----------
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "swt" / "reviews_balanced.json"
INPUT_PATH = BASE_DIR / "data" / "processed" / "reviews_clean.json"
OUTPUT_PATH = BASE_DIR / "data" / "artifacts" / "sentiment_scores.json"

# ---------- Load Training Data ----------
with open(TRAIN_PATH, "r") as f:
    train_data = json.load(f)

train_texts = [r["text"] for r in train_data]
train_labels = [r["label"] for r in train_data]  # 0=negative, 1=neutral, 2=positive

# ---------- Load Inference Data ----------
with open(INPUT_PATH, "r") as f:
    reviews = json.load(f)

texts = [r["clean_text"] for r in reviews]

print("Training samples:", len(train_texts))
print("Label distribution:", Counter(train_labels))

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)

X_train = vectorizer.fit_transform(train_texts)
X = vectorizer.transform(texts)

# ---------- Train ----------
svm = LinearSVC()
model = CalibratedClassifierCV(svm)
model.fit(X_train, train_labels)

# ---------- Predict ----------
preds = model.predict(X)
probs = model.predict_proba(X)

label_map = {0: "negative", 1: "neutral", 2: "positive"}

for i, r in enumerate(reviews):
    r["svm_label"] = label_map[preds[i]]
    r["svm_score"] = float(max(probs[i]))

# ---------- Save ----------
with open(OUTPUT_PATH, "w") as f:
    json.dump(reviews, f, indent=2)

print("SVM sentiment completed successfully.")
