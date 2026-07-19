import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import build_preprocessing_pipeline


DATA_PATH = Path(__file__).resolve().parent / "dataset" / "train_u6lujuX_CVtuZ9i.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "static" / "model_results"
REPORTS_DIR = Path(__file__).resolve().parent / "static" / "reports"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
PIPELINE_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"


def load_dataset(dataset_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset from a CSV file."""
    return pd.read_csv(dataset_path)


def prepare_data(df: pd.DataFrame):
    """Prepare training and test data using the preprocessing pipeline."""
    if "Loan_Status" not in df.columns:
        raise KeyError("The dataset must contain a Loan_Status target column.")

    df = df.copy()
    y = df["Loan_Status"].map({"N": 0, "Y": 1})
    if y.isna().any():
        raise ValueError("Loan_Status contains values outside expected ['Y', 'N'].")

    X = df.drop(columns=["Loan_Status"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    feature_names = _get_feature_names(pipeline, X_train)
    return X_train_transformed, X_test_transformed, y_train, y_test, pipeline, feature_names


def _get_feature_names(pipeline, X_sample: pd.DataFrame):
    """Extract feature names after preprocessing if available."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out(X_sample.columns)
    except Exception:
        transformed = pipeline.transform(X_sample)
        return [f"feature_{i}" for i in range(transformed.shape[1])]


def train_models(X_train, y_train):
    """Train a collection of classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models and return a comparison DataFrame."""
    records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_score = model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_score)

        print("\n" + "=" * 60)
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        records.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": roc_auc,
            }
        )

    results_df = pd.DataFrame(records)
    return results_df


def select_best_model(results_df: pd.DataFrame, models: dict):
    """Select the best model by accuracy, then by F1 score as a tiebreaker."""
    sorted_df = results_df.sort_values(["Accuracy", "F1"], ascending=[False, False])
    best_model_name = sorted_df.iloc[0]["Model"]
    return best_model_name, models[best_model_name], sorted_df


def save_best_model(best_model, pipeline):
    """Persist the best model and preprocessing pipeline to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_model, BEST_MODEL_PATH)
    dump(pipeline, PIPELINE_PATH)


def _save_figure(fig, filename: str) -> None:
    """Persist a figure to both the legacy and reports folders."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / filename, dpi=300)
    fig.savefig(REPORTS_DIR / filename, dpi=300)
    plt.close(fig)


def plot_accuracy_comparison(results_df: pd.DataFrame):
    """Plot model accuracy comparison as a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Accuracy", y="Model", data=results_df.sort_values("Accuracy", ascending=False), palette="Blues_d", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    _save_figure(fig, "model_comparison.png")


def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all trained models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_score):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_figure(fig, "roc_curves.png")


def plot_confusion_matrix(best_model, X_test, y_test):
    """Plot the confusion matrix for the best model."""
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    _save_figure(fig, "confusion_matrices.png")


def plot_feature_importance(models, feature_names):
    """Plot feature importance for the Random Forest or XGBoost model."""
    importance_model = None
    for name in ["Random Forest", "XGBoost"]:
        if name in models:
            importance_model = models[name]
            importance_name = name
            break

    if importance_model is None or not hasattr(importance_model, "feature_importances_"):
        return

    importance = importance_model.feature_importances_
    feature_names = np.array(feature_names)
    sorted_idx = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=importance[sorted_idx], y=feature_names[sorted_idx], ax=ax, palette="viridis")
    ax.set_title(f"Feature Importance ({importance_name})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    _save_figure(fig, "feature_importances.png")


def plot_precision_recall_comparison(results_df: pd.DataFrame):
    """Plot precision and recall comparison for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df = results_df.sort_values("Accuracy", ascending=False)
    index = np.arange(len(results_df))
    bar_width = 0.35

    ax.barh(index - bar_width / 2, results_df["Precision"], height=bar_width, label="Precision")
    ax.barh(index + bar_width / 2, results_df["Recall"], height=bar_width, label="Recall")
    ax.set_yticks(index)
    ax.set_yticklabels(results_df["Model"])
    ax.set_xlim(0, 1)
    ax.set_title("Precision vs Recall")
    ax.set_xlabel("Score")
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, "precision_recall_curves.png")


def plot_calibration_curve(best_model, X_test, y_test):
    """Plot the calibration curve for the best model."""
    if not hasattr(best_model, "predict_proba"):
        return

    try:
        y_score = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        return

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_score, n_bins=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, "calibration_curves.png")


def save_metrics(best_name, best_model, X_train, y_train, results_df):
    """Persist the best model metrics for the details page."""
    metrics_row = results_df.loc[results_df["Model"] == best_name].iloc[0]
    training_accuracy = round(float(best_model.score(X_train, y_train)), 4)
    testing_accuracy = round(float(metrics_row["Accuracy"]), 4)
    cross_val_accuracy = round(float(cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy").mean()), 4)

    metrics = {
        "best_model": best_name,
        "training_accuracy": training_accuracy,
        "testing_accuracy": testing_accuracy,
        "cross_validation_accuracy": cross_val_accuracy,
        "roc_auc": round(float(metrics_row["ROC-AUC"]), 4),
        "precision": round(float(metrics_row["Precision"]), 4),
        "recall": round(float(metrics_row["Recall"]), 4),
        "f1_score": round(float(metrics_row["F1"]), 4),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_results(models, best_model, X_test, y_test, results_df, feature_names):
    """Generate all model evaluation visualizations."""
    plot_accuracy_comparison(results_df)
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test)
    plot_feature_importance(models, feature_names)
    plot_precision_recall_comparison(results_df)
    plot_calibration_curve(best_model, X_test, y_test)


def main():
    """Execute the model training, evaluation, and persistence workflow."""
    try:
        sns.set_style("whitegrid")
        df = load_dataset()
        X_train, X_test, y_train, y_test, pipeline, feature_names = prepare_data(df)
        models = train_models(X_train, y_train)
        results_df = evaluate_models(models, X_test, y_test)

        print("\n" + "-" * 65)
        print("Model comparison table:")
        print(results_df.to_string(index=False, float_format="{:.4f}".format))
        print("-" * 65)

        best_name, best_model, sorted_df = select_best_model(results_df, models)
        save_best_model(best_model, pipeline)
        plot_results(models, best_model, X_test, y_test, results_df, feature_names)
        save_metrics(best_name, best_model, X_train, y_train, results_df)

        best_metrics = sorted_df.loc[sorted_df["Model"] == best_name].iloc[0]
        print("\n" + "= " * 40)
        print("SMART LENDER MODEL TRAINING COMPLETED")
        print("= " * 40)
        print(f"Best Model: {best_name}")
        print(f"Accuracy: {best_metrics['Accuracy']:.4f}")
        print(f"F1 Score: {best_metrics['F1']:.4f}")
        print("Model saved successfully.")
    except Exception as error:
        print("An error occurred during training:", str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
