# AttritionIQ — Employee Attrition Prediction System

A full-stack HR analytics web app that predicts whether an employee is likely
to leave the company, built with **Flask**, **scikit-learn (Random Forest)**,
and vanilla **HTML/CSS/JS**.

---

## 1. Project Structure

```
employee-attrition-prediction/
├── app.py                      # Flask application (routes, validation, prediction)
├── train_model.py              # Generates dataset + trains & saves model.pkl
├── model.pkl                   # Trained Random Forest pipeline (joblib)
├── wsgi.py                     # WSGI entry point (PythonAnywhere / generic hosts)
├── requirements.txt            # Pinned Python dependencies
├── Procfile                    # Gunicorn start command (Render / Railway)
├── render.yaml                 # Render.com infra-as-code (optional, one-click)
├── runtime.txt                 # Python version pin
├── .gitignore
├── data/
│   ├── sample_employee_data.csv   # Full synthetic training dataset
│   ├── sample_display.json        # 8 sample rows shown on the dashboard
│   └── model_metrics.json         # Accuracy / AUC / feature importances
├── templates/
│   ├── base.html                # Shared layout: navbar + footer
│   ├── index.html               # Home — dashboard-style landing page
│   ├── predict.html             # Prediction input form
│   ├── result.html              # Prediction result (gauge + breakdown)
│   ├── about.html                # About the project
│   ├── contact.html             # Contact form
│   ├── 404.html / 500.html      # Error pages
└── static/
    ├── style.css                # Full design system (navy / teal / clay HR theme)
    └── script.js                # Nav toggle, sliders, client-side validation
```

---

## 2. Machine Learning Model

- **Algorithm:** Random Forest Classifier (300 trees, max depth 10, balanced class weights)
- **Framework:** scikit-learn, wrapped in a `Pipeline` with a `ColumnTransformer`
  that one-hot encodes categorical fields — so `model.pkl` handles *both*
  preprocessing and prediction. `app.py` never needs to duplicate encoding logic.
- **Features (12):** Age, Gender, Department, Job Role, Monthly Income,
  Years at Company, Job Satisfaction, Work Life Balance, Overtime,
  Distance From Home, Training Times Last Year, Performance Rating.
- **Training data:** 3,000 synthetic employee records generated with
  domain-informed probability rules (overtime, low satisfaction, poor
  work-life balance, low pay, and long commutes raise attrition risk),
  calibrated to a realistic ~16–17% overall attrition rate.
- **Performance:** ~83% test accuracy, ~0.82 ROC-AUC (see `data/model_metrics.json`).

### Regenerating the model

The trained model is already included (`model.pkl`), but you can regenerate
it (e.g. with a different random seed, or your own real dataset) by running:

```bash
python train_model.py
```

This overwrites `model.pkl`, `data/sample_employee_data.csv`,
`data/sample_display.json`, and `data/model_metrics.json`.

**To use your own real HR dataset:** replace the `generate_dataset()` function
in `train_model.py` with a `pd.read_csv(...)` call on your data, keeping the
same column names listed above, then re-run the script.

---

## 3. Running Locally

### Prerequisites
- Python 3.11+ (project pinned to 3.12.3)

### Setup

```bash
# 1. Clone / unzip the project, then cd into it
cd employee-attrition-prediction

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Regenerate the model — a trained model.pkl is already included
python train_model.py

# 5. Run the development server
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

### Running with Gunicorn locally (production-style)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --threads 4
```

---

## 4. Deployment

The app is deployment-ready for **Render**, **Railway**, and
**PythonAnywhere**. It uses `PORT` from the environment (falling back to
5000 locally) and Gunicorn as the production WSGI server.

### 4.1 Render

**Option A — One-click via `render.yaml`:**
1. Push this project to a GitHub repository.
2. In the Render dashboard, choose **New → Blueprint**, and point it at your repo.
   Render will read `render.yaml` and configure everything automatically.

**Option B — Manual setup:**
1. Push the project to GitHub.
2. In Render, choose **New → Web Service** and connect the repo.
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
   - **Environment:** Python 3
4. Deploy. Render automatically injects `$PORT`.

### 4.2 Railway

1. Push the project to GitHub.
2. In Railway, choose **New Project → Deploy from GitHub repo**.
3. Railway auto-detects Python and reads the `Procfile` (`web: gunicorn app:app ...`) —
   no extra configuration needed.
4. Under **Variables**, Railway sets `$PORT` automatically; the Procfile already
   binds to it.
5. Deploy. Your app will be live at the generated `*.up.railway.app` domain.

### 4.3 PythonAnywhere

1. Upload the project (via git clone, or the Files tab / zip upload) to your
   PythonAnywhere account, e.g. `/home/<username>/employee-attrition-prediction`.
2. Open a **Bash console** and install dependencies into a virtualenv:
   ```bash
   cd employee-attrition-prediction
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Go to the **Web** tab → **Add a new web app** → choose **Manual configuration**
   (not "Flask" preset) → select the matching Python version.
4. Set the **Virtualenv** path to `/home/<username>/employee-attrition-prediction/venv`.
5. Edit the **WSGI configuration file** (linked on the Web tab) to read:
   ```python
   import sys
   project_home = '/home/<username>/employee-attrition-prediction'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)

   from wsgi import application
   ```
6. Set the **Source code** / **Working directory** to the project folder.
7. Click **Reload** on the Web tab. Your app is live at `<username>.pythonanywhere.com`.

### 4.4 Environment variables (all platforms)

| Variable     | Purpose                                  | Required |
|--------------|-------------------------------------------|----------|
| `PORT`       | Port Gunicorn binds to                    | Set automatically by Render/Railway |
| `SECRET_KEY` | Flask session secret                      | Recommended in production |

---

## 5. Application Features

- Dashboard-style homepage with live-style risk gauge, KPI stats, and a
  sample training-data table.
- 12-field employee input form with HTML5 + JavaScript client-side
  validation, mirrored by server-side validation in `app.py`.
- Prediction result page: Likely to Leave / Likely to Stay verdict,
  probability gauge, confidence score, full profile recap, and the
  model's top global risk drivers.
- Responsive layout (desktop / tablet / mobile) with a collapsible nav menu.
- About page with model metrics and methodology; Contact page with a
  validated form.
- Consistent HR-analytics visual system: deep navy, signal teal, and
  clay-red risk accent; Space Grotesk / Inter / IBM Plex Mono type.

---

## 6. Notes & Limitations

- The model is trained on **synthetic data** for demonstration purposes.
  It should not be used to make real employment or HR decisions without
  retraining on real, de-identified, consented data with proper fairness
  and bias auditing.
- The Contact form records submissions in-memory for the demo; it does not
  send real email. Wire it up to an email service (e.g. Flask-Mail, SendGrid)
  for production use.
