from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATASET_FILE = Path("dataset.csv")
MODEL_FILE = Path("model.pkl")
OUT_DIR = Path("static")


def main() -> None:
    if not DATASET_FILE.exists():
        raise FileNotFoundError("dataset.csv not found")
    if not MODEL_FILE.exists():
        raise FileNotFoundError("model.pkl not found")

    OUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATASET_FILE)
    X = df.drop(columns=["Yield"])
    y = df["Yield"]

    model_bundle = joblib.load(MODEL_FILE)
    model = model_bundle["model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # 1) Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="white", linewidth=0.5)
    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#1f6f3f", linewidth=2)
    ax.set_title("Actual vs Predicted Yield")
    ax.set_xlabel("Actual Yield (tons/hectare)")
    ax.set_ylabel("Predicted Yield (tons/hectare)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=220)
    plt.close(fig)

    # 2) Residual distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, bins=25, kde=True, ax=ax, color="#2f7d32")
    ax.axvline(0, linestyle="--", color="#8b1f1f", linewidth=1.8)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_distribution.png", dpi=220)
    plt.close(fig)

    # 3) Residuals vs Predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="white", linewidth=0.5, color="#157347")
    ax.axhline(0, linestyle="--", color="#8b1f1f", linewidth=1.8)
    ax.set_title("Residuals vs Predicted Yield")
    ax.set_xlabel("Predicted Yield (tons/hectare)")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residuals_vs_predicted.png", dpi=220)
    plt.close(fig)

    # 4) MAE by Crop (on test set)
    test_df = X_test.copy()
    test_df["Actual"] = y_test.values
    test_df["Predicted"] = y_pred
    test_df["AbsError"] = np.abs(test_df["Actual"] - test_df["Predicted"])
    crop_error = (
        test_df.groupby("Crop", as_index=False)["AbsError"]
        .mean()
        .sort_values("AbsError", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=crop_error, x="Crop", y="AbsError", ax=ax, color="#2f7d32")
    ax.set_title("Mean Absolute Error by Crop")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Mean Absolute Error")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mae_by_crop.png", dpi=220)
    plt.close(fig)

    print("Saved figures:")
    print("- static/actual_vs_predicted.png")
    print("- static/residual_distribution.png")
    print("- static/residuals_vs_predicted.png")
    print("- static/mae_by_crop.png")


if __name__ == "__main__":
    main()
