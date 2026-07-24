"""
train_model.py
----------------
Generates a realistic synthetic HR dataset and trains a Random Forest
Classifier (wrapped inside a scikit-learn Pipeline that also handles
one-hot encoding of categorical fields) to predict employee attrition.

The final trained Pipeline object is serialized to model.pkl using joblib,
so app.py can load it directly and call .predict() / .predict_proba()
on a raw pandas DataFrame without needing to re-implement preprocessing.

Run once during setup:
    python train_model.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# ----------------------------------------------------------------------
# 1. Define feature schema (kept in sync with app.py + the HTML form)
# ----------------------------------------------------------------------
DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources",
]
GENDERS = ["Male", "Female"]
OVERTIME = ["Yes", "No"]

CATEGORICAL_FEATURES = ["Gender", "Department", "JobRole", "OverTime"]
NUMERIC_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "DistanceFromHome", "TrainingTimesLastYear",
    "PerformanceRating",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ----------------------------------------------------------------------
# 2. Synthesize a realistic dataset using domain-informed probability
#    rules (mirrors well known attrition risk drivers from HR analytics
#    research: overtime, low satisfaction, poor work-life balance, low
#    pay, long commute, and short/very long tenure all raise risk).
# ----------------------------------------------------------------------
def generate_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 60, n_samples)
    gender = rng.choice(GENDERS, n_samples)
    department = rng.choice(DEPARTMENTS, n_samples, p=[0.30, 0.55, 0.15])
    job_role = rng.choice(JOB_ROLES, n_samples)
    monthly_income = rng.integers(15000, 150000, n_samples)
    years_at_company = rng.integers(0, 25, n_samples)
    job_satisfaction = rng.integers(1, 5, n_samples)          # 1 (Low) - 4 (Very High)
    work_life_balance = rng.integers(1, 5, n_samples)         # 1 (Bad) - 4 (Best)
    overtime = rng.choice(OVERTIME, n_samples, p=[0.30, 0.70])
    distance_from_home = rng.integers(1, 40, n_samples)
    training_times_last_year = rng.integers(0, 7, n_samples)
    performance_rating = rng.integers(1, 5, n_samples)        # 1 (Low) - 4 (Outstanding)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Department": department,
        "JobRole": job_role,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime,
        "DistanceFromHome": distance_from_home,
        "TrainingTimesLastYear": training_times_last_year,
        "PerformanceRating": performance_rating,
    })

    # ---- Rule-based latent attrition probability ----
    risk = np.zeros(n_samples)

    risk += (overtime == "Yes") * 0.28
    risk += (5 - job_satisfaction) * 0.09
    risk += (5 - work_life_balance) * 0.08
    risk += np.clip((35000 - monthly_income) / 35000, 0, 1) * 0.22
    risk += np.clip((distance_from_home - 10) / 30, 0, 1) * 0.12
    risk += np.where(years_at_company <= 1, 0.18, 0.0)
    risk += np.where((years_at_company >= 2) & (years_at_company <= 4), 0.06, 0.0)
    risk += np.where(training_times_last_year == 0, 0.07, 0.0)
    risk += np.where(performance_rating <= 1, 0.05, 0.0)
    risk += np.where(age < 25, 0.08, 0.0)
    risk += np.where(department == "Sales", 0.05, 0.0)

    # small random noise so the boundary isn't perfectly deterministic
    risk += rng.normal(0, 0.05, n_samples)
    # sigmoid calibrated (empirically) to produce a realistic ~17% overall
    # attrition rate, matching typical industry HR benchmarks
    prob_leave = 1 / (1 + np.exp(-7 * (risk - 0.95)))

    attrition = rng.binomial(1, np.clip(prob_leave, 0.01, 0.95))
    df["Attrition"] = attrition  # 1 = Yes (leaves), 0 = No (stays)

    return df


# ----------------------------------------------------------------------
# 3. Build preprocessing + model pipeline
# ----------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def main():
    print("Generating synthetic employee dataset...")
    df = generate_dataset(n_samples=3000)
    df.to_csv("data/sample_employee_data.csv", index=False)
    print(f"Dataset saved to data/sample_employee_data.csv ({len(df)} rows)")
    print(f"Attrition rate: {df['Attrition'].mean():.2%}")

    X = df[ALL_FEATURES]
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest Classifier...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

    joblib.dump(pipeline, "model.pkl")
    print("\nModel saved to model.pkl")

    # Save a small sample of employee records for the dashboard's
    # "sample employee data" table
    sample = df.sample(8, random_state=1).reset_index(drop=True)
    sample.to_json("data/sample_display.json", orient="records")
    print("Sample display data saved to data/sample_display.json")

    # Save headline metrics + top feature importances for the dashboard
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    friendly = {
        "num__Age": "Age", "num__MonthlyIncome": "Monthly Income",
        "num__YearsAtCompany": "Years at Company", "num__JobSatisfaction": "Job Satisfaction",
        "num__WorkLifeBalance": "Work Life Balance", "num__DistanceFromHome": "Distance From Home",
        "num__TrainingTimesLastYear": "Training Times Last Year", "num__PerformanceRating": "Performance Rating",
    }
    imp_pairs = []
    grouped = {}
    for name, score in zip(feature_names, importances):
        if name in friendly:
            label = friendly[name]
        elif name.startswith("cat__OverTime"):
            label = "Overtime"
        elif name.startswith("cat__Gender"):
            label = "Gender"
        elif name.startswith("cat__Department"):
            label = "Department"
        elif name.startswith("cat__JobRole"):
            label = "Job Role"
        else:
            label = name
        grouped[label] = grouped.get(label, 0.0) + float(score)

    imp_pairs = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in imp_pairs)
    top_factors = [
        {"feature": k, "importance": round((v / total) * 100, 1)}
        for k, v in imp_pairs[:6]
    ]

    metrics = {
        "accuracy": round(acc * 100, 1),
        "roc_auc": round(auc * 100, 1),
        "dataset_size": int(len(df)),
        "attrition_rate": round(float(df["Attrition"].mean()) * 100, 1),
        "n_estimators": int(pipeline.named_steps["classifier"].n_estimators),
        "top_factors": top_factors,
    }
    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Model metrics saved to data/model_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
