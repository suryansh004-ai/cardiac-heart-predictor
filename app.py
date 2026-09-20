"""
Heart Disease Prediction System - Flask Web Application
======================================================
A clean, beginner-friendly Flask application providing an educational
Heart Disease Prediction interface powered by a trained Logistic Regression model.

Routes:
- GET  /         : Home / landing page with project overview and disclaimer
- GET  /predict  : Input form for patient clinical parameters with demo quick-fills
- POST /predict  : Form handler that validates, scales, predicts, and renders result
- GET  /about    : Project explanation, dataset background, and model evaluation metrics
"""

from pathlib import Path
import joblib
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "heart-disease-prediction-secret-key-for-session-messages"

# Set file paths relative to project root (no hardcoded absolute paths)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_disease_model.pkl"

# Global references for loaded artifacts
model = None
comparison_model = None
scaler = None
feature_names = []
lr_metrics = {}
rf_metrics = {}


def load_model_bundle():
    """Load the trained model and preprocessing pipeline from disk."""
    global model, comparison_model, scaler, feature_names, lr_metrics, rf_metrics
    if not MODEL_PATH.exists():
        print(f"Warning: Model file not found at {MODEL_PATH}.")
        print("Please run `python train_model.py` first to train and save the model.")
        return False

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    comparison_model = bundle.get("comparison_model")
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    lr_metrics = bundle.get("lr_metrics", {})
    rf_metrics = bundle.get("rf_metrics", {})
    print(" Model and preprocessing pipeline loaded successfully.")
    return True


# Load model at startup
load_model_bundle()


@app.route("/")
def home():
    """Home page displaying project mission, workflow overview, and disclaimers."""
    return render_template(
        "index.html",
        model_ready=(model is not None),
        lr_metrics=lr_metrics,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Render prediction form (GET) or process clinical inputs and render prediction result (POST).
    """
    if request.method == "GET":
        return render_template("predict.html", model_ready=(model is not None))

    # Ensure model is available
    if model is None or scaler is None:
        flash("Trained model is not available. Please run `python train_model.py` first.", "danger")
        return redirect(url_for("predict"))

    # Extract and validate form inputs
    try:
        # Form field extraction
        age = float(request.form.get("age", 0))
        sex = int(request.form.get("sex", 0))
        cp = int(request.form.get("cp", 0))
        trestbps = float(request.form.get("trestbps", 0))
        chol = float(request.form.get("chol", 0))
        fbs = int(request.form.get("fbs", 0))
        restecg = int(request.form.get("restecg", 0))
        thalach = float(request.form.get("thalach", 0))
        exang = int(request.form.get("exang", 0))
        oldpeak = float(request.form.get("oldpeak", 0.0))
        slope = int(request.form.get("slope", 0))
        ca = int(request.form.get("ca", 0))
        thal = int(request.form.get("thal", 2))

        # Basic range validation
        if not (1 <= age <= 120):
            raise ValueError("Age must be between 1 and 120 years.")
        if not (50 <= trestbps <= 260):
            raise ValueError("Resting Blood Pressure must be between 50 and 260 mm Hg.")
        if not (80 <= chol <= 700):
            raise ValueError("Cholesterol level must be between 80 and 700 mg/dl.")
        if not (40 <= thalach <= 250):
            raise ValueError("Maximum Heart Rate must be between 40 and 250 bpm.")
        if not (0.0 <= oldpeak <= 10.0):
            raise ValueError("ST Depression (oldpeak) must be between 0.0 and 10.0.")

        # Organize features in exact DataFrame format with column names
        import pandas as pd
        input_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=feature_names
        )

        # Preprocess features using the exact StandardScaler fitted during training
        scaled_features = scaler.transform(input_df)

        # Generate model prediction (0 = Lower Risk, 1 = Higher Likelihood)
        prediction = int(model.predict(scaled_features)[0])

        # Generate model-predicted probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        prob_risk = round(float(probabilities[1]) * 100, 1)
        prob_healthy = round(float(probabilities[0]) * 100, 1)

        # Identify key educational risk factors present in the patient profile
        risk_factors = []
        if trestbps >= 140:
            risk_factors.append(f"Elevated Resting Blood Pressure ({int(trestbps)} mm Hg - Stage 2 Hypertension range)")
        if chol >= 240:
            risk_factors.append(f"High Serum Cholesterol ({int(chol)} mg/dl - above desirable 200 mg/dl threshold)")
        if fbs == 1:
            risk_factors.append("Elevated Fasting Blood Sugar (> 120 mg/dl)")
        if exang == 1:
            risk_factors.append("Presence of Exercise-Induced Angina (Chest discomfort during exertion)")
        if oldpeak >= 1.5:
            risk_factors.append(f"Significant ST Depression ({oldpeak} mm induced during exercise)")
        if ca > 0:
            risk_factors.append(f"{ca} Major Blood Vessel(s) colored by fluoroscopy")
        if cp in [1, 2, 3]:
            chest_pain_labels = {1: "Atypical Angina", 2: "Non-anginal Chest Pain", 3: "Asymptomatic Chest Pain"}
            risk_factors.append(f"Chest Pain Type: {chest_pain_labels.get(cp)}")

        # Clean display dictionary of the patient inputs
        user_data = {
            "Age": f"{int(age)} years",
            "Sex": "Male" if sex == 1 else "Female",
            "Chest Pain Type": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}.get(cp, cp),
            "Resting Blood Pressure": f"{int(trestbps)} mm Hg",
            "Serum Cholesterol": f"{int(chol)} mg/dl",
            "Fasting Blood Sugar > 120 mg/dl": "True" if fbs == 1 else "False",
            "Resting ECG": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}.get(restecg, restecg),
            "Max Heart Rate": f"{int(thalach)} bpm",
            "Exercise Angina": "Yes" if exang == 1 else "No",
            "ST Depression (oldpeak)": f"{oldpeak}",
            "ST Slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}.get(slope, slope),
            "Major Vessels (ca)": f"{ca}",
            "Thalassemia": {1: "Fixed Defect", 2: "Normal Blood Flow", 3: "Reversible Defect"}.get(thal, thal),
        }

        return render_template(
            "result.html",
            prediction=prediction,
            prob_risk=prob_risk,
            prob_healthy=prob_healthy,
            risk_factors=risk_factors,
            user_data=user_data,
        )

    except ValueError as ve:
        flash(f"Input validation error: {str(ve)}", "danger")
        return redirect(url_for("predict"))
    except Exception as e:
        flash(f"An unexpected error occurred while processing prediction: {str(e)}", "danger")
        return redirect(url_for("predict"))


@app.route("/about")
def about():
    """About page explaining dataset, algorithms, preprocessing, and evaluation metrics."""
    return render_template(
        "about.html",
        lr_metrics=lr_metrics,
        rf_metrics=rf_metrics,
    )


if __name__ == "__main__":
    # Host on 127.0.0.1 port 5000 for local development
    print("Starting Flask web server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
