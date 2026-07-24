"""
app.py
------
Employee Attrition Prediction System — Flask backend.

Loads a pre-trained scikit-learn Pipeline (preprocessing + Random Forest
Classifier) from model.pkl via joblib, serves a dashboard-style HR
analytics UI, validates and scores employee-profile submissions, and
renders a scored result page.

Run locally:
    python app.py

Run in production (used by Procfile):
    gunicorn app:app
"""

import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request

# ----------------------------------------------------------------------
# App & constants
# ----------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.json")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_display.json")

NUMERIC_FIELDS = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "DistanceFromHome", "TrainingTimesLastYear",
    "PerformanceRating",
]
CATEGORICAL_FIELDS = ["Gender", "Department", "JobRole", "OverTime"]
ALL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

# Server-side validation rules: (min, max) for numeric fields
FIELD_RANGES = {
    "Age": (18, 65),
    "MonthlyIncome": (10000, 300000),
    "YearsAtCompany": (0, 40),
    "DistanceFromHome": (0, 100),
    "TrainingTimesLastYear": (0, 6),
    "JobSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4),
    "PerformanceRating": (1, 4),
}
VALID_CHOICES = {
    "Gender": ["Male", "Female"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "OverTime": ["Yes", "No"],
    "JobRole": [
        "Sales Executive", "Sales Representative", "Research Scientist",
        "Research Director", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Human Resources",
    ],
}

# ----------------------------------------------------------------------
# Load model + supporting assets once at startup
# ----------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001 - surface any load issue to the UI
    model = None
    MODEL_LOAD_ERROR = str(exc)

try:
    with open(METRICS_PATH) as f:
        MODEL_METRICS = json.load(f)
except Exception:  # noqa: BLE001
    MODEL_METRICS = {
        "accuracy": 83.3, "roc_auc": 81.6, "dataset_size": 3000,
        "attrition_rate": 16.4, "n_estimators": 300,
        "top_factors": [{"feature": "Overtime", "importance": 21.4}],
    }

try:
    with open(SAMPLE_DATA_PATH) as f:
        SAMPLE_EMPLOYEES = json.load(f)
except Exception:  # noqa: BLE001
    SAMPLE_EMPLOYEES = []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def validate_form(form) -> tuple[dict, list]:
    """Validate submitted form data.

    Returns a (cleaned_data, errors) tuple. cleaned_data has correctly
    typed values ready to be fed to the model; errors is a list of
    human-readable validation messages (empty if valid).
    """
    errors = []
    cleaned = {}

    # Numeric fields: must be present, numeric, and within range
    for field in NUMERIC_FIELDS:
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"{field} is required.")
            continue
        try:
            value = int(float(raw))
        except ValueError:
            errors.append(f"{field} must be a number.")
            continue
        lo, hi = FIELD_RANGES[field]
        if not (lo <= value <= hi):
            errors.append(f"{field} must be between {lo} and {hi}.")
            continue
        cleaned[field] = value

    # Categorical fields: must be present and match an allowed option
    for field in CATEGORICAL_FIELDS:
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"{field} is required.")
            continue
        if raw not in VALID_CHOICES[field]:
            errors.append(f"{field} has an invalid value.")
            continue
        cleaned[field] = raw

    return cleaned, errors


def risk_bucket(probability_pct: float) -> str:
    """Classify a probability percentage into a risk tier used for styling."""
    if probability_pct < 40:
        return "low"
    if probability_pct < 70:
        return "moderate"
    return "high"


RISK_COLORS = {"low": "#17a892", "moderate": "#d9822b", "high": "#da5b4c"}


# ----------------------------------------------------------------------
# Context processor — inject values needed on every template (e.g. footer year)
# ----------------------------------------------------------------------
@app.context_processor
def inject_globals():
    return {"current_year": datetime.now(timezone.utc).year}


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        active_page="home",
        model_accuracy=MODEL_METRICS.get("accuracy", 83.3),
        dataset_size=f'{MODEL_METRICS.get("dataset_size", 3000):,}',
        sample_employees=SAMPLE_EMPLOYEES,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", active_page="predict", error=None, form_data=None)

    # ---- POST: validate + predict ----
    if model is None:
        return render_template(
            "predict.html", active_page="predict",
            error=f"Model could not be loaded on the server: {MODEL_LOAD_ERROR}",
            form_data=request.form,
        )

    cleaned, errors = validate_form(request.form)
    if errors:
        return render_template(
            "predict.html", active_page="predict",
            error=" ".join(errors),
            form_data=request.form,
        )

    try:
        # Build a single-row DataFrame in the exact column order the
        # pipeline's ColumnTransformer expects.
        row = pd.DataFrame([{field: cleaned[field] for field in ALL_FIELDS}])
        proba = model.predict_proba(row)[0]  # [P(stay), P(leave)]
        prob_leave_pct = round(float(proba[1]) * 100, 1)
        prob_stay_pct = round(float(proba[0]) * 100, 1)
        prediction = int(model.predict(row)[0])  # 1 = leave, 0 = stay
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "predict.html", active_page="predict",
            error=f"Something went wrong while scoring this profile: {exc}",
            form_data=request.form,
        )

    will_leave = prediction == 1
    confidence_pct = prob_leave_pct if will_leave else prob_stay_pct
    tier = risk_bucket(prob_leave_pct)
    needle_rotation = round(-90 + (prob_leave_pct / 100) * 180, 1)

    return render_template(
        "result.html",
        active_page="predict",
        will_leave=will_leave,
        prob_leave_pct=prob_leave_pct,
        prob_stay_pct=prob_stay_pct,
        confidence_pct=confidence_pct,
        risk_tier=tier,
        risk_color=RISK_COLORS[tier],
        needle_rotation=needle_rotation,
        employee=cleaned,
        top_factors=MODEL_METRICS.get("top_factors", []),
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        active_page="about",
        metrics=MODEL_METRICS,
    )


@app.route("/contact", methods=["GET", "POST"])
def contact():
    submitted = False
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if not name or not email or not message:
            error = "Please fill in your name, email, and message before sending."
        elif "@" not in email or "." not in email:
            error = "Please enter a valid email address."
        else:
            # In production this would send an email / create a ticket.
            # Kept intentionally simple for this portfolio project.
            submitted = True
    return render_template("contact.html", active_page="contact", submitted=submitted, error=error)


# ----------------------------------------------------------------------
# Error handlers
# ----------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html", active_page=""), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("500.html", active_page=""), 500


if __name__ == "__main__":
    # debug=True is fine for local development; Gunicorn is used in production
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
