# Pipeline nnu vecha sentiment link cheyyan use cheyyunneyaa

from ai.supervised_sentiment.infer import load_sentiment_model, predict_sentiment
from ai.fusion.combine_scores import fuse_predictions

""" server start aayi kazhinjee distilbert load cheyyuvol, ella requestlum cheyyula - 
    model functionte porath ee load cheyuvol bcs analyse() ullil cheythal,
    ella requestmm wait cheyyum for time buffer, whereas
    topll load cheythal ella calls nm ready aarkm """



_tokenizer = None
_model = None

def analyse(text: str) -> dict:
    global _tokenizer, _model

    """ full 2 stage pipeline nte function aan
        final label, confidence, decided_by return chyym """

    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_sentiment_model()

    # 1. Distilbert
    distilbert_result = predict_sentiment(text, _tokenizer, _model)

    # 2. KMeans
    final_result = fuse_predictions(distilbert_result, text)
    return final_result
