

import os
import joblib
import pandas as pd

# Always resolve pipeline.pkl relative to this file, regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "discount",
    "price_diff",
    "desc_length",
    "review_sentiment",
]

# Load once at import time
_pipeline = joblib.load(os.path.join(_HERE, "pipeline.pkl"))


def predict(data: dict) -> float:

    df = pd.DataFrame([data])[FEATURES]

    # Continuous sales score prediction
    return float(_pipeline.predict(df)[0])


if __name__ == "__main__":

    sample = {
        "discounted_price":   499,
        "actual_price":       999,
        "discount_percentage": 50,
        "rating":              3.5,
        "discount":            500,
        "price_diff":          500,
        "desc_length":         80,
        "review_sentiment":    0.2,
    }

    result = predict(sample)

    
    if result < 40:
        label = "Low Selling"
    elif result < 70:
        label = "Moderate Sales"
    else:
        label = "High Sales"

    print(f"Sales Score Prediction: {result:.2f}")
    print(f"Category: {label}")
