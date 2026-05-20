import re
import pandas as pd
import numpy as np
from textblob import TextBlob

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

df = pd.read_csv("amazon_clean_data.csv")


drop_cols = ["user_id", "user_name", "review_id", "img_link",
             "product_link", "product_id"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

df["discount_percentage"] = (
    df["discount_percentage"].astype(str).str.replace("%", "").astype(float)
)
df["rating_count"].fillna(df["rating_count"].median(), inplace=True)

 
df["low_selling"] = (df["rating_count"] < df["rating_count"].median()).astype(int)

df["price_diff"] = df["actual_price"] - df["discounted_price"]
df["desc_length"] = df["about_product"].apply(lambda x: len(str(x).split()))
df["review_sentiment"] = df["review_content"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)


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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print("=" * 55)
print("Regression Report")
print("=" * 55)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

joblib.dump(pipeline, "pipeline.pkl")
print("\nPipeline saved → pipeline.pkl")
print(f"Input features expected: {FEATURES}")
