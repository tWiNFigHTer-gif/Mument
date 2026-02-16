# Inference script
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import torch.nn.functional as F

# Logic: Load the model from your artifact folder onto your RTX GPU
script_dir = Path(__file__).parent
MODEL_PATH = str(script_dir / "../../data/artifacts/sentiment_model")

# Label mapping for our ternary classifier
LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

def load_sentiment_model():
    print("⏳ Loading model to GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to('cuda')
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    """Run inference on a single review"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    return {
        'label': LABEL_MAP[predicted_class],
        'score': confidence,
        'class_id': predicted_class
    }

if __name__ == "__main__":
    tokenizer, model = load_sentiment_model()
    
      # Test cases matching EV review domain
    test_reviews = [
        "The EV is fantastic, I love the range and acceleration",  # Should be POSITIVE
        "Terrible charging infrastructure, battery drains too fast",  # Should be NEGATIVE  
        "The car is decent for daily commuting, nothing special"  # Should be NEUTRAL
    ]
    
    print("\n--- Model Test Results ---")
    for review in test_reviews:
        result = predict_sentiment(review, tokenizer, model)
        print(f"Review: {review}")
        print(f"Result: {result['label']} (Confidence: {result['score']:.4f})\n")