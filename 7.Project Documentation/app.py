import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, url_for
from joblib import dump, load

from preprocessing import create_fitted_pipeline


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "dataset" / "train_u6lujuX_CVtuZ9i.csv"
HISTORY_FILE = BASE_DIR / "history.json"
BATCH_RESULTS_FILE = BASE_DIR / "static" / "batch_predictions.csv"
REPORTS_DIR = BASE_DIR / "static" / "reports"
LEGACY_REPORTS_DIR = BASE_DIR / "static" / "model_results"

FORM_FIELDS = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term",
    "credit_history",
    "property_area",
]

TRAIN_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

REQUIRED_BATCH_COLUMNS = TRAIN_COLUMNS
NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _artifact_error_message() -> str:
    return (
        "The prediction model artifacts are missing or incompatible. "
        "Please ensure the model files exist in the models folder."
    )


def load_artifacts():
    """Load the trained model and preprocessing pipeline with graceful fallbacks."""
    preferred_model = MODELS_DIR / "best_model.joblib"
    fallback_model = MODELS_DIR / "loan_eligibility_model.joblib"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"

    model_path = preferred_model if preferred_model.exists() else fallback_model if fallback_model.exists() else None

    if model_path is None:
        raise FileNotFoundError(_artifact_error_message())

    if not pipeline_path.exists():
        pipeline = create_fitted_pipeline(DATASET_PATH)
        dump(pipeline, pipeline_path)
    else:
        pipeline = load(pipeline_path)

    model = load(model_path)
    return model, pipeline


def prepare_input(form_data: dict) -> pd.DataFrame:
    """Convert HTML form data into a DataFrame for prediction."""
    data = {
        "Gender": [str(form_data.get("gender") or "Unknown")],
        "Married": [str(form_data.get("married") or "No")],
        "Dependents": [str(form_data.get("dependents") or "0")],
        "Education": [str(form_data.get("education") or "Not Graduate")],
        "Self_Employed": [str(form_data.get("self_employed") or "No")],
        "ApplicantIncome": [_safe_float(form_data.get("applicant_income"), 0.0)],
        "CoapplicantIncome": [_safe_float(form_data.get("coapplicant_income"), 0.0)],
        "LoanAmount": [_safe_float(form_data.get("loan_amount"), 0.0)],
        "Loan_Amount_Term": [_safe_float(form_data.get("loan_term"), 360.0)],
        "Credit_History": [_safe_float(form_data.get("credit_history"), 1.0)],
        "Property_Area": [str(form_data.get("property_area") or "Urban")],
    }
    return pd.DataFrame(data, columns=TRAIN_COLUMNS)


def prepare_batch_input(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a batch CSV for prediction using the same column order as training."""
    missing_columns = [column for column in REQUIRED_BATCH_COLUMNS if column not in batch_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    prepared = batch_df.loc[:, REQUIRED_BATCH_COLUMNS].copy()
    for column in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        prepared[column] = prepared[column].fillna("Unknown").astype(str)
    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
    return prepared


def generate_prediction(input_df: pd.DataFrame):
    """Run prediction using the loaded model and preprocessing pipeline."""
    model, pipeline = load_artifacts()
    transformed = pipeline.transform(input_df)

    if getattr(model, "n_features_in_", None) is not None and transformed.shape[1] != int(model.n_features_in_):
        raise ValueError(
            f"Feature mismatch: the model expects {model.n_features_in_} features but the preprocessing pipeline produced {transformed.shape[1]}."
        )

    prediction_value = int(model.predict(transformed)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(transformed)[0][1])
    return "Eligible" if prediction_value == 1 else "Not Eligible", probability


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return []


def _static_rel_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(BASE_DIR / "static").as_posix()
    except ValueError:
        return path.name


def _resolve_report_image(*candidates: str) -> Path | None:
    for directory in [REPORTS_DIR, LEGACY_REPORTS_DIR]:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path
    return None


def ensure_report_assets() -> dict:
    """Generate the report images when missing and return their static-relative paths."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    required_files = [
        "model_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "feature_importances.png",
        "precision_recall_curves.png",
        "calibration_curves.png",
    ]
    if not metrics_path.exists() or not all((REPORTS_DIR / filename).exists() for filename in required_files):
        try:
            subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], cwd=str(BASE_DIR), check=True, capture_output=True, text=True)
        except Exception as exc:
            print(f"Report generation failed: {exc}")

    return {
        "model_comparison": _static_rel_path(_resolve_report_image("model_comparison.png", "model_accuracy_comparison.png")),
        "confusion_matrix": _static_rel_path(_resolve_report_image("confusion_matrices.png", "best_model_confusion_matrix.png")),
        "roc_curve": _static_rel_path(_resolve_report_image("roc_curves.png", "roc_curve_comparison.png")),
        "precision_recall": _static_rel_path(_resolve_report_image("precision_recall_curves.png", "precision_vs_recall_comparison.png")),
        "feature_importance": _static_rel_path(_resolve_report_image("feature_importances.png", "feature_importance.png")),
        "calibration_curve": _static_rel_path(_resolve_report_image("calibration_curves.png")),
    }


def load_report_metrics() -> dict:
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_history(records):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def append_history_record(form_data: dict, prediction: str, probability: float | None):
    history = load_history()
    record = {
        "prediction_id": f"APP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "applicant_name": str(form_data.get("applicant_name") or "N/A"),
        "prediction": prediction,
        "probability": round(float(probability), 2) if probability is not None else None,
        "loan_amount": str(form_data.get("loan_amount") or "0"),
        "income": str(form_data.get("applicant_income") or "0"),
        "status": prediction,
    }
    history.append(record)
    save_history(history)
    return record


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return application()
    return redirect(url_for("application"))


@app.route("/application", methods=["GET", "POST"])
def application():
    prediction = None
    probability = None
    form_data = {}
    error_message = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        missing_fields = [field for field in FORM_FIELDS if not str(form_data.get(field, "")).strip()]
        if missing_fields:
            error_message = f"Please complete all required fields before predicting. Missing: {', '.join(missing_fields)}"
        else:
            try:
                input_df = prepare_input(request.form)
                prediction, probability = generate_prediction(input_df)
                append_history_record(form_data, prediction, probability)
            except FileNotFoundError as exc:
                error_message = str(exc)
                traceback.print_exc()
            except ValueError as exc:
                error_message = f"Prediction failed: {exc}"
                traceback.print_exc()
            except Exception as exc:
                error_message = f"Prediction failed unexpectedly: {exc}"
                traceback.print_exc()

    return render_template(
        "application.html",
        active_page="application",
        prediction=prediction,
        probability=round(float(probability), 2) if probability is not None else None,
        form_data=form_data,
        error_message=error_message,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    batch_summary = None
    batch_results = []
    error_message = None
    batch_download_url = None

    if request.method == "POST":
        uploaded_file = request.files.get("batch_file")
        if uploaded_file is None or uploaded_file.filename == "":
            error_message = "Please upload a CSV file before running batch predictions."
        else:
            try:
                batch_df = pd.read_csv(uploaded_file)
                prepared_df = prepare_batch_input(batch_df)
                model, pipeline = load_artifacts()
                transformed = pipeline.transform(prepared_df)
                predictions = model.predict(transformed)
                probabilities = model.predict_proba(transformed)[:, 1] if hasattr(model, "predict_proba") else [None] * len(prepared_df)

                result_df = prepared_df.copy()
                result_df["Prediction"] = ["Eligible" if value == 1 else "Not Eligible" for value in predictions]
                result_df["Probability"] = [round(float(prob), 2) if prob is not None else None for prob in probabilities]
                result_df.to_csv(BATCH_RESULTS_FILE, index=False)
                batch_results = result_df.to_dict(orient="records")
                batch_summary = {
                    "total": int(len(result_df)),
                    "approved": int((result_df["Prediction"] == "Eligible").sum()),
                    "rejected": int((result_df["Prediction"] == "Not Eligible").sum()),
                    "approval_rate": round((result_df["Prediction"].eq("Eligible").mean() * 100), 2),
                }
                batch_download_url = url_for("batch_download")
            except FileNotFoundError as exc:
                error_message = str(exc)
                traceback.print_exc()
            except ValueError as exc:
                error_message = f"Batch prediction failed: {exc}"
                traceback.print_exc()
            except Exception as exc:
                error_message = f"Batch prediction failed unexpectedly: {exc}"
                traceback.print_exc()

    return render_template(
        "batch.html",
        active_page="batch",
        batch_summary=batch_summary,
        batch_results=batch_results,
        error_message=error_message,
        batch_download_url=batch_download_url,
    )


@app.route("/batch/download")
def batch_download():
    if not BATCH_RESULTS_FILE.exists():
        return redirect(url_for("batch"))
    return send_file(BATCH_RESULTS_FILE, mimetype="text/csv", as_attachment=True, download_name="batch_predictions.csv")


@app.route("/history", methods=["GET"])
def history():
    history_records = load_history()
    return render_template("history.html", active_page="history", history_records=history_records)


@app.route("/history/export/csv")
def history_export_csv():
    history_records = load_history()
    if not history_records:
        return redirect(url_for("history"))

    export_df = pd.DataFrame(history_records)
    export_path = BASE_DIR / "static" / "history_export.csv"
    export_df.to_csv(export_path, index=False)
    return send_file(export_path, mimetype="text/csv", as_attachment=True, download_name="prediction_history.csv")


@app.route("/reports", methods=["GET"])
def reports():
    history_records = load_history()
    report_stats = {
        "total": len(history_records),
        "approved": sum(1 for record in history_records if record.get("prediction") == "Eligible"),
        "rejected": sum(1 for record in history_records if record.get("prediction") == "Not Eligible"),
    }
    report_stats["approval_rate"] = round((report_stats["approved"] / report_stats["total"] * 100), 2) if report_stats["total"] else 0.0
    return render_template("reports.html", active_page="reports", report_stats=report_stats)


@app.route("/model", methods=["GET"])
def model():
    sample_models = [
        {
            "name": "Logistic Regression",
            "slug": "logistic-regression",
            "accuracy": 86.2,
            "precision": 84.7,
            "recall": 82.3,
            "f1_score": 83.5,
            "roc_auc": 0.89,
        },
        {
            "name": "Decision Tree",
            "slug": "decision-tree",
            "accuracy": 82.1,
            "precision": 80.4,
            "recall": 79.0,
            "f1_score": 79.7,
            "roc_auc": 0.85,
        },
        {
            "name": "Random Forest",
            "slug": "random-forest",
            "accuracy": 88.4,
            "precision": 86.8,
            "recall": 85.5,
            "f1_score": 86.1,
            "roc_auc": 0.92,
        },
        {
            "name": "KNN",
            "slug": "knn",
            "accuracy": 78.6,
            "precision": 76.9,
            "recall": 75.2,
            "f1_score": 76.0,
            "roc_auc": 0.81,
        },
        {
            "name": "XGBoost",
            "slug": "xgboost",
            "accuracy": 89.3,
            "precision": 88.1,
            "recall": 87.0,
            "f1_score": 87.5,
            "roc_auc": 0.93,
        },
    ]
    return render_template("model.html", active_page="model", models=sample_models)


@app.route("/model/<model_slug>", methods=["GET"])
def model_details(model_slug: str):
    details = {
        "logistic-regression": {
            "name": "Logistic Regression",
            "description": "A linear classification model suitable for binary decisions and risk scoring.",
            "accuracy": 86.2,
            "precision": 84.7,
            "recall": 82.3,
            "f1_score": 83.5,
            "roc_auc": 0.89,
            "overview": "Logistic Regression models the probability of eligibility using log odds and is interpretable for credit decisions.",
            "advantages": [
                "Fast training and low complexity.",
                "Interpretable coefficients.",
                "Works well with linearly separable data.",
            ],
            "disadvantages": "Can underperform on highly non-linear relationships.",
        },
        "decision-tree": {
            "name": "Decision Tree",
            "description": "A branching model that learns segmented borrower profiles and decision rules.",
            "accuracy": 82.1,
            "precision": 80.4,
            "recall": 79.0,
            "f1_score": 79.7,
            "roc_auc": 0.85,
            "overview": "Decision Trees split borrower data into branches, making the underwriting logic easy to visualize.",
            "advantages": [
                "Simple decision rules.",
                "Handles non-linear patterns.",
                "No need for feature scaling.",
            ],
            "disadvantages": "Prone to overfitting without pruning.",
        },
        "random-forest": {
            "name": "Random Forest",
            "description": "An ensemble of decision trees that improves underwriting robustness and accuracy.",
            "accuracy": 88.4,
            "precision": 86.8,
            "recall": 85.5,
            "f1_score": 86.1,
            "roc_auc": 0.92,
            "overview": "Random Forest aggregates multiple trees to reduce variance and strengthen loan approval predictions.",
            "advantages": [
                "High accuracy and stability.",
                "Handles complex relationships.",
                "Resistant to overfitting.",
            ],
            "disadvantages": "Less interpretable than single trees.",
        },
        "knn": {
            "name": "KNN",
            "description": "A proximity-based model that classifies borrowers based on nearest neighbors.",
            "accuracy": 78.6,
            "precision": 76.9,
            "recall": 75.2,
            "f1_score": 76.0,
            "roc_auc": 0.81,
            "overview": "KNN compares each applicant to similar past borrowers to make eligibility decisions.",
            "advantages": [
                "Simple and intuitive.",
                "Non-parametric approach.",
            ],
            "disadvantages": "Slow prediction on large datasets.",
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "A high-performance gradient boosting model optimized for underwriting accuracy.",
            "accuracy": 89.3,
            "precision": 88.1,
            "recall": 87.0,
            "f1_score": 87.5,
            "roc_auc": 0.93,
            "overview": "XGBoost uses gradient boosting to build strong sequential learners for loan approval scoring.",
            "advantages": [
                "Excellent predictive performance.",
                "Handles missing data well.",
                "Highly tunable.",
            ],
            "disadvantages": "Can be resource intensive.",
        },
    }
    model = details.get(model_slug)
    if model is None:
        return redirect(url_for("model"))

    report_assets = ensure_report_assets()
    metrics = load_report_metrics()
    model.update(
        {
            "model_name": model["name"],
            "selected_best_model": metrics.get("best_model", model["name"]),
            "training_accuracy": metrics.get("training_accuracy", model["accuracy"]),
            "testing_accuracy": metrics.get("testing_accuracy", model["accuracy"]),
            "cross_validation_accuracy": metrics.get("cross_validation_accuracy", model["accuracy"]),
            "roc_auc_score": metrics.get("roc_auc", model["roc_auc"]),
            "precision": metrics.get("precision", model["precision"]),
            "recall": metrics.get("recall", model["recall"]),
            "f1_score": metrics.get("f1_score", model["f1_score"]),
            "charts": report_assets,
        }
    )
    return render_template("model_details.html", active_page="model", model=model)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
