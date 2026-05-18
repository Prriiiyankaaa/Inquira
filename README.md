# Inquira 🔍

**Inquira** is an AI-powered product sales predictor built with Flask. It analyzes product pricing, discount metrics, review sentiment, and product descriptions to predict whether a product will have **Good Sales** or **Low Selling** performance — and if underperforming, explains *why* using SHAP and gives actionable suggestions to improve.

---

## Features

- **Sales Prediction** — Classifies a product as "Good Sales" or "Low Selling" using a trained ML pipeline.
- **SHAP Explainability** — For low-selling predictions, surfaces the top contributing factors driving the result.
- **Actionable Suggestions** — Provides concrete recommendations to improve product performance.
- **NLP Description Analysis** — Analyzes product descriptions using TextBlob and Gensim to identify language quality issues.
- **Flask Web Interface** — Simple, clean web UI for entering product details and viewing results.

---

## Project Structure

```
Inquira/
├── app.py                   # Flask app — routes and request handling
├── predictor.py             # Loads pipeline.pkl and runs predictions
├── explain.py               # SHAP-based explanation and suggestions
├── description_analyzer.py  # NLP analysis of product descriptions
├── train_pipeline.py        # Script to train and save the ML pipeline
├── pipeline.pkl             # Trained scikit-learn pipeline (serialized)
├── amazon_clean_data.csv    # Cleaned Amazon product dataset
├── sales.ipynb              # Exploratory data analysis notebook
├── style.css                # Stylesheet
├── templates/               # Jinja2 HTML templates
└── requirements.txt         # Python dependencies
```

---

## Input Features

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| `discounted_price`   | Current selling price after discount             |
| `actual_price`       | Original MRP of the product                     |
| `discount_percentage`| Percentage discount applied                      |
| `rating`             | Average customer rating                          |
| `discount`           | Absolute discount amount                         |
| `price_diff`         | Difference between actual and discounted price  |
| `description`        | Product description text (auto-computes length) |
| `review_sentiment`   | Sentiment score derived from customer reviews   |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Prriiiyankaaa/Inquira.git
cd Inquira
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

```bash
python train_pipeline.py
```

> Skip this step if you want to use the pre-trained `pipeline.pkl` included in the repo.

### 4. Run the app

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`.

---

## Tech Stack

| Layer        | Tools                                        |
|--------------|----------------------------------------------|
| Web Framework| Flask                                        |
| ML           | scikit-learn, SHAP, joblib                   |
| NLP          | TextBlob, Gensim                             |
| Data         | pandas, numpy                                |
| Visualization| matplotlib, seaborn                         |
| Dataset      | Amazon product data (`amazon_clean_data.csv`)|

---

## How It Works

1. The user submits product details via the web form.
2. `predictor.py` loads the serialized `pipeline.pkl` and returns a prediction.
3. If the product is predicted as **Low Selling**:
   - `explain.py` uses SHAP to identify which features are hurting performance and generates suggestions.
   - `description_analyzer.py` runs NLP analysis on the product description to flag quality issues.
4. Results — prediction, SHAP reasons, suggestions, and description analysis — are rendered in the UI.

---

## Dataset

The project uses `amazon_clean_data.csv`, a cleaned dataset of Amazon product listings with fields including pricing, ratings, discounts, and review data. The notebook `sales.ipynb` contains the full EDA and feature engineering process.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## Author

**Prriiiyankaaa** — [GitHub](https://github.com/Prriiiyankaaa)
