from flask import Flask, request, render_template
from predictor import predict
from explain import explain_prediction
from description_analyzer import analyze
import pandas as pd

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip) 

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        description = request.form.get("description", "").strip()

        # Auto-compute desc_length from the actual description text
        desc_length = len(description.split()) if description else 0

        data = {
            "discounted_price":    float(request.form["discounted_price"]),
            "actual_price":        float(request.form["actual_price"]),
            "discount_percentage": float(request.form["discount_percentage"]),
            "rating":              float(request.form["rating"]),
            "discount":            float(request.form["discount"]),
            "price_diff":          float(request.form["price_diff"]),
            "desc_length":         float(desc_length),
            "review_sentiment":    float(request.form["review_sentiment"]),
        }

       
        prediction = predict(data)

        shap_reasons = []
        shap_suggestions = []
        if prediction == 1:
            df = pd.DataFrame([data])
            shap_reasons, shap_suggestions = explain_prediction(df)

        #Description using NLP
        desc_analysis = analyze(description)

        return render_template(
            "index.html",
            prediction      = "Low Selling" if prediction == 1 else "Good Sales",
            shap_reasons    = shap_reasons,
            shap_suggestions= shap_suggestions,
            desc_analysis   = desc_analysis,
            description     = description,
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

