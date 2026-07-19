from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "train_u6lujuX_CVtuZ9i.csv"


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the loan dataset from disk."""
    return pd.read_csv(dataset_path)


def remove_loan_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the Loan_ID identifier before preprocessing or model training."""
    return df.drop(columns=["Loan_ID"], errors="ignore")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features while preventing division by zero."""
    engineered = df.copy()

    engineered["TotalIncome"] = (
        engineered["ApplicantIncome"] + engineered["CoapplicantIncome"]
    )

    epsilon = np.finfo(float).eps
    loan_amount = engineered["LoanAmount"].replace(0, np.nan)
    loan_term = engineered["Loan_Amount_Term"].replace(0, np.nan)

    engineered["EMI"] = engineered["LoanAmount"] / (loan_term + epsilon)
    engineered["IncomePerLoan"] = engineered["TotalIncome"] / (loan_amount + epsilon)

    return engineered


def build_preprocessing_pipeline() -> Pipeline:
    """Create and return a reusable preprocessing pipeline."""
    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    numeric_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "TotalIncome",
        "EMI",
        "IncomePerLoan",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("remove_id", FunctionTransformer(remove_loan_id, validate=False)),
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def preprocess_input(input_data: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Transform raw input data using a fitted preprocessing pipeline."""
    return pipeline.transform(input_data)


def create_fitted_pipeline(dataset_path: Path = DATASET_PATH) -> Pipeline:
    """Load training data, fit the preprocessing pipeline, and return it."""
    df = load_dataset(dataset_path)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(df)
    return pipeline
