import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
DATASET_PATH = Path("dataset.csv")
MODEL_PATH = Path("model.pkl")
STATIC_DIR = Path("static")

REQUIRED_COLUMNS = [
    "State",
    "District",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production",
    "Rainfall",
    "Temperature",
    "Humidity",
    "Soil_Type",
    "Fertilizer_Usage",
    "Yield",
]


def create_synthetic_dataset(file_path: Path, n_samples: int = 1500) -> None:
    """Create a realistic synthetic crop dataset when a real dataset is unavailable."""
    rng = np.random.default_rng(RANDOM_STATE)

    states = ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu"]
    districts = [
        "Ludhiana",
        "Karnal",
        "Kanpur",
        "Nashik",
        "Mysuru",
        "Coimbatore",
        "Amritsar",
        "Nagpur",
    ]
    seasons = ["Kharif", "Rabi", "Zaid"]
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"]
    soil_types = ["Alluvial", "Black", "Red", "Laterite", "Clay", "Sandy"]

    crop_base_yield = {
        "Rice": 3.1,
        "Wheat": 3.6,
        "Maize": 2.8,
        "Cotton": 2.1,
        "Sugarcane": 5.2,
        "Soybean": 2.4,
    }

    season_adjustment = {"Kharif": 0.4, "Rabi": 0.6, "Zaid": 0.1}

    rows = []
    for _ in range(n_samples):
        state = rng.choice(states)
        district = rng.choice(districts)
        crop_year = int(rng.integers(2005, 2024))
        season = rng.choice(seasons)
        crop = rng.choice(crops)
        area = float(np.round(rng.uniform(5.0, 250.0), 2))

        rainfall = float(np.round(rng.normal(980, 210), 2))
        temperature = float(np.round(rng.normal(27, 4), 2))
        humidity = float(np.round(rng.uniform(40, 95), 2))
        soil_type = rng.choice(soil_types)
        fertilizer = float(np.round(rng.uniform(40, 340), 2))

        base = crop_base_yield[crop] + season_adjustment[season]
        weather_gain = 0.0012 * rainfall - 0.032 * abs(temperature - 27) + 0.008 * (humidity - 60)
        fert_gain = 0.0045 * fertilizer
        soil_gain = 0.28 if soil_type in {"Alluvial", "Black", "Clay"} else 0.08

        state_noise = {
            "Punjab": 0.35,
            "Haryana": 0.25,
            "Uttar Pradesh": 0.15,
            "Maharashtra": 0.05,
            "Karnataka": 0.18,
            "Tamil Nadu": 0.22,
        }[state]

        yield_value = base + weather_gain + fert_gain + soil_gain + state_noise + rng.normal(0, 0.35)
        yield_value = float(np.round(max(0.8, yield_value), 3))

        production = float(np.round(max(5.0, area * yield_value + rng.normal(0, 12)), 3))

        rows.append(
            {
                "State": state,
                "District": district,
                "Crop_Year": crop_year,
                "Season": season,
                "Crop": crop,
                "Area": area,
                "Production": production,
                "Rainfall": rainfall,
                "Temperature": temperature,
                "Humidity": humidity,
                "Soil_Type": soil_type,
                "Fertilizer_Usage": fertilizer,
                "Yield": yield_value,
            }
        )

    synthetic_df = pd.DataFrame(rows)

    # Add a small amount of missing data so missing-value handling is demonstrated.
    for column in ["Rainfall", "Temperature", "Soil_Type", "Fertilizer_Usage"]:
        missing_idx = synthetic_df.sample(frac=0.02, random_state=RANDOM_STATE).index
        synthetic_df.loc[missing_idx, column] = np.nan

    synthetic_df.to_csv(file_path, index=False)
    print(f"Synthetic dataset created at {file_path.resolve()}")


def load_or_create_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV or create one if it does not exist."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        create_synthetic_dataset(file_path)

    df = pd.read_csv(file_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create preprocessing pipelines for numerical and categorical features."""
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor, numeric_columns, categorical_columns


def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    """Save model performance charts as an image."""
    STATIC_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(
        data=results_df,
        x="Model",
        y="R2",
        hue="Model",
        ax=axes[0],
        palette="Greens_d",
        legend=False,
    )
    axes[0].set_title("R2 Score by Model")
    axes[0].set_ylim(0, max(1.0, results_df["R2"].max() + 0.05))
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=results_df,
        x="Model",
        y="RMSE",
        hue="Model",
        ax=axes[1],
        palette="YlGn",
        legend=False,
    )
    axes[1].set_title("RMSE by Model")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    chart_path = STATIC_DIR / "model_comparison.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    print(f"Saved model comparison chart: {chart_path}")


def save_feature_importance_chart(
    best_pipeline: Pipeline,
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Save feature importance chart using direct importance or permutation importance."""
    STATIC_DIR.mkdir(exist_ok=True)
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    feature_importance_df: pd.DataFrame

    if hasattr(model, "feature_importances_"):
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        onehot_features = onehot.get_feature_names_out(categorical_columns).tolist()
        transformed_feature_names = numeric_columns + onehot_features
        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame(
            {"Feature": transformed_feature_names, "Importance": importances}
        )
    else:
        # Fallback for models like SVR where direct feature importance is unavailable.
        permutation = permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            n_repeats=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        feature_importance_df = pd.DataFrame(
            {"Feature": X_test.columns, "Importance": permutation.importances_mean}
        )

    top_features = feature_importance_df.sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="Greens",
        legend=False,
    )
    plt.title(f"Top Features ({best_model_name})")
    plt.tight_layout()

    chart_path = STATIC_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()
    print(f"Saved feature importance chart: {chart_path}")


def train_and_evaluate() -> None:
    """Train all required models, compare metrics, save the best model."""
    df = load_or_create_dataset(DATASET_PATH)
    print(f"Dataset loaded with shape: {df.shape}")

    X = df.drop(columns=["Yield"])
    y = df["Yield"]

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "AdaBoostRegressor": AdaBoostRegressor(
            n_estimators=250,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(C=60, epsilon=0.1, gamma="scale", kernel="rbf"),
    }

    results = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        results.append(
            {
                "Model": model_name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )
        trained_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    print("\nModel Performance:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.loc[0, "Model"]
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(
        {
            "model": best_pipeline,
            "model_name": best_model_name,
            "feature_columns": X.columns.tolist(),
        },
        MODEL_PATH,
    )
    print(f"\nBest model ({best_model_name}) saved as: {MODEL_PATH.resolve()}")

    results_df.to_csv("model_metrics.csv", index=False)
    print("Saved model metrics to model_metrics.csv")

    save_model_comparison_chart(results_df)
    save_feature_importance_chart(
        best_pipeline,
        best_model_name,
        X_test,
        y_test,
        numeric_columns,
        categorical_columns,
    )


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as exc:
        print(f"Error during training: {exc}")
