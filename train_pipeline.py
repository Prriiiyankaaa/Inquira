"""
train_pipeline.py
-----------------
Builds a clean sklearn Pipeline on 8 raw product features,
evaluates it, and saves the pipeline to pipeline.pkl.

Features used (all numeric, no manual pre-scaling needed outside the pipe):
    discounted_price, actual_price, discount_percentage, rating,
    discount, price_diff, desc_length, review_sentiment
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import joblib

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("amazon_clean_data.csv")

# ── 2. Drop columns not used for modelling ───────────────────────────────────
drop_cols = ["user_id", "user_name", "review_id", "img_link",
             "product_link", "product_id"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 3. Fix dtypes ─────────────────────────────────────────────────────────────
df["discount_percentage"] = (
    df["discount_percentage"].astype(str).str.replace("%", "").astype(float)
)
df["rating_count"].fillna(df["rating_count"].median(), inplace=True)

# ── 4. Target ─────────────────────────────────────────────────────────────────
df["low_selling"] = (df["rating_count"] < df["rating_count"].median()).astype(int)

# ── 5. Engineered features ────────────────────────────────────────────────────
df["price_diff"] = df["actual_price"] - df["discounted_price"]
df["desc_length"] = df["about_product"].apply(lambda x: len(str(x).split()))
df["review_sentiment"] = df["review_content"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

# ── 6. Select the 8 raw input features ───────────────────────────────────────
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

X = df[FEATURES]
y = df["low_selling"]

# ── 7. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 8. Pipeline ───────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("=" * 55)
print("Classification Report")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Good Sales", "Low Selling"]))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

# ── 10. Save ──────────────────────────────────────────────────────────────────
joblib.dump(pipeline, "pipeline.pkl")
print("\nPipeline saved → pipeline.pkl")
print(f"Input features expected: {FEATURES}")
