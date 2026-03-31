"""
predictor.py
------------
Loads the saved pipeline and exposes a single `predict(data)` function.
Accepts a dict of 8 raw product features; returns 0 (Good Sales) or 1 (Low Selling).
"""

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


def predict(data: dict) -> int:
    """
    Parameters
    ----------
    data : dict
        Must contain all 8 keys listed in FEATURES.

    Returns
    -------
    int
        1 → Low Selling   |   0 → Good Sales
    """
    df = pd.DataFrame([data])[FEATURES]
    return int(_pipeline.predict(df)[0])


# ── Quick smoke-test ──────────────────────────────────────────────────────────
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
    label  = "Low Selling" if result == 1 else "Good Sales"
    print(f"Prediction: {result}  →  {label}")
