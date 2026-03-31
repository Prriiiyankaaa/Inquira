"""
explain.py
----------
SHAP-based explainability for the RandomForest inside the pipeline.

Public API
----------
explain_prediction(df)  →  (reasons: list[str], suggestions: list[str])
    • df        : pd.DataFrame with 1 row, same 8 feature columns as training
    • reasons   : top-3 human-readable sentences explaining *why* the product is low-selling
    • suggestions: one actionable fix per reason
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────
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


_LABELS = {
    "discounted_price":   "Discounted Price",
    "actual_price":       "Actual / MRP",
    "discount_percentage":"Discount Percentage",
    "rating":             "Product Rating",
    "discount":           "Absolute Discount (₹)",
    "price_diff":         "Price Gap (MRP − Selling)",
    "desc_length":        "Description Length (words)",
    "review_sentiment":   "Review Sentiment",
}

# Direction-aware reason templates
# called as _REASON[feat](shap_val, raw_val)
_REASON = {
    "discounted_price":    lambda s, v: (
        f"The selling price (₹{v:.0f}) is high relative to similar products, pushing buyers away."
        if s > 0 else
        f"The selling price (₹{v:.0f}) is very low, which may signal poor quality to buyers."
    ),
    "actual_price":        lambda s, v: (
        f"The MRP (₹{v:.0f}) is steep; even with a discount the product feels expensive."
        if s > 0 else
        f"A low MRP (₹{v:.0f}) limits the perceived value of any discount offered."
    ),
    "discount_percentage": lambda s, v: (
        f"The discount is only {v:.0f}%, which is too small to attract deal-seeking shoppers."
        if s > 0 else
        f"An aggressive {v:.0f}% discount is unusual and may raise authenticity concerns."
    ),
    "rating":              lambda s, v: (
        f"A rating of {v:.1f} is below the 4.0 threshold most customers expect before buying."
        if s > 0 else
        f"A strong rating of {v:.1f} is a positive signal — this feature is helping sales."
    ),
    "discount":            lambda s, v: (
        f"The absolute discount (₹{v:.0f}) is modest; buyers may not feel the deal is worthwhile."
        if s > 0 else
        f"A large absolute discount (₹{v:.0f}) is attractive but may seem inflated."
    ),
    "price_diff":          lambda s, v: (
        f"The ₹{v:.0f} gap between MRP and selling price looks artificially inflated, eroding trust."
        if s > 0 else
        f"A small price gap (₹{v:.0f}) gives buyers confidence the deal is genuine."
    ),
    "desc_length":         lambda s, v: (
        f"The description is only {v:.0f} words — too thin to convey value and build buyer confidence."
        if s > 0 else
        f"A {v:.0f}-word description may be overwhelming; consider making it more concise."
    ),
    "review_sentiment":    lambda s, v: (
        f"Review sentiment ({v:.2f}) skews negative, signalling poor post-purchase experience."
        if s > 0 else
        f"Review sentiment ({v:.2f}) is positive — reviews are working in this product's favour."
    ),
}

# Actionable suggestion per feature
_SUGGESTION = {
    "discounted_price":    "Reprice to be within 10–15% of top competitors in the same category.",
    "actual_price":        "Consider repositioning in a lower price tier or bundling to justify the MRP.",
    "discount_percentage": "Raise the discount to 30–40% to meet shopper expectations on deal platforms.",
    "rating":              "Address the top complaints in 1-star reviews and follow up with buyers to improve the rating above 4.0.",
    "discount":            "Offer a minimum ₹200–300 flat discount or add a 'Limited Deal' badge to increase urgency.",
    "price_diff":          "Align MRP and selling price more honestly; inflated strikethrough prices reduce buyer trust.",
    "desc_length":         "Rewrite the description: aim for 80–120 words, lead with the top 3 benefits, and add bullet points.",
    "review_sentiment":    "Reply to negative reviews publicly, and send a post-purchase email asking satisfied customers for feedback.",
}
#loading pipeline
_pipeline = joblib.load(os.path.join(_HERE, "pipeline.pkl"))
_scaler   = _pipeline.named_steps["scaler"]
_rf       = _pipeline.named_steps["rf"]


_explainer = shap.TreeExplainer(_rf)



def explain_prediction(df: pd.DataFrame):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Single-row DataFrame with the 8 model features.

    Returns
    -------
    reasons     : list[str]  – top-3 human-readable causes of low-selling
    suggestions : list[str]  – one actionable fix per reason
    """
    df = df[FEATURES].copy()

    # Scale with the same scaler used during training
    X_scaled = _scaler.transform(df)

    # SHAP values — shape: (n_classes, n_samples, n_features) for RF
    shap_values = _explainer.shap_values(X_scaled)

    # Class-1 (Low Selling) SHAP values for the single sample
    if isinstance(shap_values, list):
        # sklearn-style: list[class0_array, class1_array]
        sv_class1 = np.array(shap_values[1]).flatten()
    else:
        # newer SHAP returns (n_samples, n_features, n_classes)
        sv_class1 = np.array(shap_values).reshape(-1, len(FEATURES), 2)[0, :, 1]

    raw_values = df.iloc[0]  # original (unscaled) values for readable output

    # Rank features by absolute SHAP contribution to Low Selling (class 1)
    ranked = sorted(
        zip(FEATURES, sv_class1),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    reasons     = []
    suggestions = []

    for feat, shap_val in ranked[:3]:
        raw_val = raw_values[feat]
        reasons.append(_REASON[feat](shap_val, raw_val))
        suggestions.append(_SUGGESTION[feat])

    return reasons, suggestions


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "discounted_price":    499,
        "actual_price":        999,
        "discount_percentage":  50,
        "rating":               3.5,
        "discount":             500,
        "price_diff":           500,
        "desc_length":           80,
        "review_sentiment":      0.2,
    }
    df_sample = pd.DataFrame([sample])
    reasons, suggestions = explain_prediction(df_sample)

    print("\n── Top-3 reasons this product may be low-selling ──")
    for i, (r, s) in enumerate(zip(reasons, suggestions), 1):
        print(f"\n[{i}] Reason    : {r}")
        print(f"    Suggestion: {s}")
