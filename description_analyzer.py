"""
description_analyzer.py
-----------------------
Pure NLP analysis of a product description string.
No ML model required — rule-based + TextBlob sentiment.

Returns a structured dict with:
  - issues       : list of specific problems found
  - suggestions  : matching actionable fixes
  - missing      : key selling elements absent from the description
  - score        : 0–100 quality score
  - tone         : 'Positive' | 'Neutral' | 'Negative'
"""

import re
from textblob import TextBlob


# ── Keyword catalogs ──────────────────────────────────────────────────────────

_SPEC_PATTERNS = [
    r'\d+\s*(gb|tb|mb|ghz|mhz|mp|w|v|hz|kg|g|cm|mm|m|inch|\")',
    r'\d+\s*-?\s*(bit|core|port|pin|amp|mah|rpm)',
    r'\d+\s*(x|×)\s*\d+',        # dimensions like 10x5
]

_TRUST_KEYWORDS = [
    'warranty', 'guarantee', 'certified', 'original', 'genuine',
    'brand', 'iso', 'tested', 'quality', 'approved', 'authentic'
]

_FEATURE_KEYWORDS = [
    'feature', 'material', 'compatible', 'support', 'includes',
    'built', 'design', 'technology', 'performance', 'durable',
    'lightweight', 'portable', 'waterproof', 'wireless', 'fast',
    'easy', 'safe', 'efficient', 'powerful', 'premium', 'heavy duty'
]

_OFFER_KEYWORDS = [
    'free', 'bonus', 'combo', 'bundle', 'offer', 'deal', 'discount',
    'limited', 'exclusive', 'pack', 'set', 'kit', 'gift'
]

_SOCIAL_PROOF_KEYWORDS = [
    'customers', 'rated', 'reviews', 'trusted', 'popular', 'bestseller',
    'top rated', 'recommended', 'award', 'million', 'thousand', 'users'
]

_USE_CASE_KEYWORDS = [
    'ideal for', 'perfect for', 'suitable for', 'designed for',
    'great for', 'used for', 'best for', 'works with', 'use case'
]

_WEAK_WORDS = [
    'good', 'nice', 'great product', 'very good', 'best product',
    'amazing product', 'wonderful', 'awesome product'
]


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze(description: str) -> dict:
    """
    Parameters
    ----------
    description : str   Raw product description text.

    Returns
    -------
    dict with keys: issues, suggestions, missing, score, tone, word_count
    """
    if not description or not description.strip():
        return {
            "issues":      ["No description provided."],
            "suggestions": ["Add a detailed product description (80–150 words)."],
            "missing":     [],
            "score":       0,
            "tone":        "Neutral",
            "word_count":  0,
        }

    text       = description.strip()
    text_lower = text.lower()
    words      = text.split()
    word_count = len(words)
    sentences  = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    issues      = []
    suggestions = []
    missing     = []
    score       = 100   # deduct for each problem found

    # ── 1. Length ─────────────────────────────────────────────────────────────
    if word_count < 30:
        issues.append(f"Description is very short ({word_count} words) — buyers won't trust a sparse listing.")
        suggestions.append("Expand to at least 80 words. Cover what the product is, what it does, and who it's for.")
        score -= 20
    elif word_count < 60:
        issues.append(f"Description is brief ({word_count} words) — not enough detail to convince buyers.")
        suggestions.append("Aim for 80–150 words. Add specs, benefits, and a use-case sentence.")
        score -= 10
    elif word_count > 300:
        issues.append(f"Description is too long ({word_count} words) — buyers lose interest quickly.")
        suggestions.append("Trim to under 200 words. Lead with the most important benefits first.")
        score -= 8

    # ── 2. Technical specifications ───────────────────────────────────────────
    has_specs = any(re.search(p, text_lower) for p in _SPEC_PATTERNS)
    if not has_specs:
        missing.append("Technical specifications (dimensions, weight, capacity, speed, etc.)")
        issues.append("No technical specs mentioned — shoppers compare numbers before buying.")
        suggestions.append("Add at least 2–3 measurable specs (e.g. '10,000 mAh', '1.5 kg', '6-inch display').")
        score -= 15

    # ── 3. Warranty / Trust signals ───────────────────────────────────────────
    has_trust = any(kw in text_lower for kw in _TRUST_KEYWORDS)
    if not has_trust:
        missing.append("Warranty or trust signal (e.g. '1-year warranty', 'ISI certified')")
        issues.append("No warranty or trust signal found — this is a top reason buyers hesitate.")
        suggestions.append("Mention warranty duration, certifications, or quality assurance ('1-year brand warranty', 'BIS certified').")
        score -= 15

    # ── 4. Features / Benefits ────────────────────────────────────────────────
    has_features = any(kw in text_lower for kw in _FEATURE_KEYWORDS)
    if not has_features:
        missing.append("Feature highlights or benefits")
        issues.append("No product features or benefits are called out explicitly.")
        suggestions.append("List 3–5 key features as bullet points (e.g. 'Anti-slip grip', 'Fast charge support', 'Dust resistant').")
        score -= 12

    # ── 5. Use-case / Target audience ────────────────────────────────────────
    has_use_case = any(kw in text_lower for kw in _USE_CASE_KEYWORDS)
    if not has_use_case:
        missing.append("Use-case or target audience ('Ideal for...', 'Perfect for...')")
        issues.append("Description doesn't say who this product is for or when to use it.")
        suggestions.append("Add a sentence like 'Ideal for students and remote workers' or 'Perfect for daily commutes'.")
        score -= 8

    # ── 6. Offers / Bundle / Value ────────────────────────────────────────────
    has_offer = any(kw in text_lower for kw in _OFFER_KEYWORDS)
    if not has_offer:
        missing.append("Value-add mention (free accessories, combo, kit, etc.)")
        score -= 5

    # ── 7. Social proof ───────────────────────────────────────────────────────
    has_social = any(kw in text_lower for kw in _SOCIAL_PROOF_KEYWORDS)
    if not has_social:
        missing.append("Social proof ('Trusted by 10,000+ customers', 'Top rated', etc.)")
        score -= 5

    # ── 8. Weak / vague language ─────────────────────────────────────────────
    weak_found = [w for w in _WEAK_WORDS if w in text_lower]
    if weak_found:
        quoted = ", ".join('"' + w + '"' for w in weak_found[:3])
        issues.append(f"Vague filler phrases detected: {quoted}.")
        suggestions.append("Replace vague adjectives with specific claims (e.g. instead of 'great product', write '4.5-star rated by 2,000 buyers').")
        score -= 8

    # ── 9. Sentence structure ────────────────────────────────────────────────
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len > 30:
            issues.append("Sentences are very long and hard to scan on a mobile screen.")
            suggestions.append("Break long sentences into shorter ones (10–20 words). Use bullet points for features.")
            score -= 5

    # ── 10. Tone / Sentiment ─────────────────────────────────────────────────
    blob      = TextBlob(text)
    polarity  = blob.sentiment.polarity
    if polarity > 0.2:
        tone = "Positive"
    elif polarity < -0.1:
        tone = "Negative"
        issues.append("Description has a negative tone — this can deter buyers subconsciously.")
        suggestions.append("Rewrite in a positive, benefit-focused tone. Avoid words like 'not', 'no', 'problem', 'issue'.")
        score -= 10
    else:
        tone = "Neutral"
        if word_count > 40:   # short neutral is just low-info
            issues.append("Description tone is flat and unenthusiastic — it doesn't make the product appealing.")
            suggestions.append("Use energetic, benefit-driven language: 'Stay connected all day with a 10,000 mAh battery' instead of 'has 10,000 mAh battery'.")
            score -= 5

    # ── Features to add suggestions ──────────────────────────────────────────
    features_to_add = []
    if not has_specs:
        features_to_add.append("Technical specs (size, weight, capacity, speed)")
    if not has_trust:
        features_to_add.append("Warranty / certification details")
    if not has_use_case:
        features_to_add.append("Target audience / ideal use-case sentence")
    if not has_offer:
        features_to_add.append("Package contents or bundled accessories")
    if not has_social:
        features_to_add.append("Customer count or trust badge")

    # Deduplicate and pair issues with suggestions
    paired = list(dict.fromkeys(zip(issues, suggestions)))   # preserve order, remove dupes
    final_issues      = [p[0] for p in paired]
    final_suggestions = [p[1] for p in paired]

    return {
        "issues":          final_issues,
        "suggestions":     final_suggestions,
        "missing":         missing,
        "features_to_add": features_to_add,
        "score":           max(0, score),
        "tone":            tone,
        "word_count":      word_count,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "Good product. Nice quality. Very useful for home use."
    result = analyze(sample)
    print(f"Score : {result['score']}/100  |  Tone: {result['tone']}  |  Words: {result['word_count']}\n")
    for i, (issue, sug) in enumerate(zip(result["issues"], result["suggestions"]), 1):
        print(f"[{i}] ⚠ {issue}")
        print(f"    → {sug}\n")
    if result["features_to_add"]:
        print("💡 Features/offers to add to description:")
        for f in result["features_to_add"]:
            print(f"   • {f}")
