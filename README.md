# Crop Yield Prediction Using Machine Learning

This project predicts crop yield (tons/hectare) from agricultural, weather, and soil factors using multiple machine learning models.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask
- Joblib

## Project Structure

- app.py
- train_model.py
- model.pkl
- dataset.csv
- templates/index.html
- static/style.css
- static/model_comparison.png
- static/feature_importance.png
- requirements.txt
- README.md

## Features

- End-to-end ML workflow:
  - load dataset
  - handle missing values
  - encode categorical data
  - scale numerical features
  - train-test split (80:20)
  - train and compare 4 regressors
- Models included:
  - Random Forest Regressor
  - AdaBoost Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
- Model evaluation metrics:
  - R2 Score
  - MAE
  - RMSE
- Automatic best-model selection and saving to `model.pkl`
- Synthetic data generation if `dataset.csv` is unavailable
- Charts:
  - model comparison chart
  - feature importance chart
- Flask web app with responsive UI, prediction form, and error handling
- User registration and login system with role-based access
- Admin dashboard to review registered users and prediction history

## Setup and Run

1. Open this project folder in VS Code.
2. Create and activate a virtual environment.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Train models and generate artifacts:

```powershell
python train_model.py
```

This step creates:
- `dataset.csv` (if missing)
- `model.pkl`
- `model_metrics.csv`
- `static/model_comparison.png`
- `static/feature_importance.png`

5. Start the Flask app:

```powershell
python app.py
```

6. Open browser:

```text
http://127.0.0.1:5000
```

## Notes

- If `model.pkl` is missing, run `python train_model.py` first.
- You can replace `dataset.csv` with your real data (same column names) and retrain.
- Unit convention used in this project:
  - `Area` is in `hectare`
  - `Yield` and predicted output are in `tons/hectare`

## Authentication

- Register a normal account from the Register page.
- Register an admin account by selecting `Admin` and entering the admin invite code.
- Default admin account:
  - Email: `admin@cropyield.local`
  - Password: `Admin@12345`
- Admin invite code:
  - `CROP-ADMIN-2026`
