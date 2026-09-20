"""
Heart Disease Prediction System - Model Training & Evaluation Pipeline
======================================================================
This script loads the UCI Heart Disease dataset, performs exploratory data analysis,
scales numerical features, trains a Logistic Regression model (with Random Forest
comparison), evaluates metrics on an unseen test set, saves visualizations to static/images/,
and exports the trained model and scaler to model/heart_disease_model.pkl.

Author: Surya
Project: heart-disease-prediction
"""

import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set paths relative to project root (no hardcoded absolute paths)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart_disease.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "heart_disease_model.pkl"
IMAGES_DIR = BASE_DIR / "static" / "images"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filepath: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total rows (samples): {df.shape[0]}")
    print(f"Total columns (features + target): {df.shape[1]}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target column: {df.columns[-1]}")
    print(f"Missing values count: {df.isnull().sum().sum()}")
    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: Path):
    """Generate and save EDA plots for documentation and web interface."""
    print("\n" + "=" * 60)
    print("2. GENERATING EXPLORATORY DATA ANALYSIS (EDA) PLOTS")
    print("=" * 60)

    # Style settings
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: Target class distribution
    plt.figure(figsize=(7, 5))
    target_counts = df["target"].value_counts()
    colors = ["#10b981", "#ef4444"]
    ax = sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        hue=target_counts.index,
        palette=colors,
        legend=False,
    )
    plt.title("Heart Disease Target Distribution (0 = Low Risk, 1 = High Risk)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Heart Disease Presence (Target)", fontsize=11)
    plt.ylabel("Number of Patients", fontsize=11)
    plt.xticks([0, 1], ["0: Lower Risk / No Disease", "1: Higher Likelihood"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    target_plot_path = output_dir / "target_distribution.png"
    plt.savefig(target_plot_path, dpi=200)
    plt.close()
    print(f" Saved target distribution plot to: {target_plot_path.name}")

    # Plot 2: Correlation matrix heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9}
    )
    plt.title("Feature Correlation Matrix (UCI Cleveland Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_plot_path = output_dir / "correlation_matrix.png"
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f" Saved correlation matrix plot to: {corr_plot_path.name}")


def train_and_evaluate(df: pd.DataFrame, output_dir: Path):
    """
    Split data, scale features using training stats only,
    train Logistic Regression & Random Forest, and evaluate on test set.
    """
    print("\n" + "=" * 60)
    print("3. MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = list(X.columns)

    # Stratified Train-Test Split (80% Train, 20% Test)
    # Stratify ensures both sets preserve the 0/1 target proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # IMPORTANT: Fit StandardScaler ONLY on X_train to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Primary Model: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics for Logistic Regression
    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "precision": float(precision_score(y_test, y_pred_lr)),
        "recall": float(recall_score(y_test, y_pred_lr)),
        "f1": float(f1_score(y_test, y_pred_lr)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    }

    # 2. Comparison Model: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # Tree-based models don't require feature scaling
    y_pred_rf = rf_model.predict(X_test)

    rf_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "precision": float(precision_score(y_test, y_pred_rf)),
        "recall": float(recall_score(y_test, y_pred_rf)),
        "f1": float(f1_score(y_test, y_pred_rf)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }

    # Print results table
    print("\n" + "-" * 60)
    print(f"{'Metric':<20} | {'Logistic Regression':<20} | {'Random Forest':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<20} | {lr_metrics['accuracy']*100:<19.2f}% | {rf_metrics['accuracy']*100:.2f}%")
    print(f"{'Precision':<20} | {lr_metrics['precision']*100:<19.2f}% | {rf_metrics['precision']*100:.2f}%")
    print(f"{'Recall':<20} | {lr_metrics['recall']*100:<19.2f}% | {rf_metrics['recall']*100:.2f}%")
    print(f"{'F1-Score':<20} | {lr_metrics['f1']*100:<19.2f}% | {rf_metrics['f1']*100:.2f}%")
    print("-" * 60)

    # Plot Confusion Matrix for Primary Model (Logistic Regression)
    cm = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: 0 (No Disease)", "Predicted: 1 (At Risk)"],
        yticklabels=["Actual: 0 (No Disease)", "Actual: 1 (At Risk)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Logistic Regression (Test Set)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\n Saved confusion matrix plot to: {cm_plot_path.name}")

    # Package model bundle
    bundle = {
        "model": lr_model,
        "comparison_model": rf_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "test_sample_count": len(y_test),
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f" Successfully saved trained model bundle to: {MODEL_PATH}")
    return bundle


def main():
    print("\nStarting Heart Disease Model Training Pipeline...")
    df = load_data(DATA_PATH)
    generate_eda_plots(df, IMAGES_DIR)
    train_and_evaluate(df, IMAGES_DIR)
    print("\n Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
