import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_output_directory(output_path: Path) -> None:
    """Create the output directory for EDA images if it does not exist."""
    output_path.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the loan dataset from a CSV file."""
    return pd.read_csv(dataset_path)


def display_dataset_overview(df: pd.DataFrame) -> None:
    """Display dataset structure, sample rows, and summary statistics."""
    print("Dataset shape:", df.shape)
    print("\nDataset information:")
    df.info()
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nStatistical summary:")
    print(df.describe(include='all').transpose())
    print("\nMissing values by column:")
    print(df.isna().sum())
    print("\nDuplicate records count:")
    print(df.duplicated().sum())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values safely before visualization."""
    cleaned = df.copy()
    # Numeric features filled with median to preserve distribution
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    # Categorical features filled with explicit missing category
    categorical_cols = cleaned.select_dtypes(include=["string", "object"]).columns
    for column in categorical_cols:
        cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save and close a Matplotlib figure to free memory."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {output_path.name}")


def plot_countplot(df: pd.DataFrame, column: str, output_path: Path, title: str, hue: str = None) -> None:
    """Create a count plot for a categorical feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Count")
    if hue:
        ax.legend(title=hue)
    save_figure(fig, output_path)


def plot_distribution(df: pd.DataFrame, column: str, output_path: Path, title: str) -> None:
    """Plot a distribution for a numeric feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True, ax=ax, color="#3478b5")
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Frequency")
    save_figure(fig, output_path)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a correlation heatmap for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    save_figure(fig, output_path)


def plot_pairplot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a pairplot for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pairplot_fig = sns.pairplot(df[numeric_cols], corner=True)
    pairplot_fig.fig.suptitle("Pairplot of Numerical Features", y=1.02)
    pairplot_fig.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(pairplot_fig.fig)
    print(f"Saved visualization: {output_path.name}")


def plot_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    """Plot boxplots for ApplicantIncome and LoanAmount."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df, x="ApplicantIncome", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Applicant Income Boxplot")
    sns.boxplot(data=df, x="LoanAmount", ax=axes[1], color="#55a868")
    axes[1].set_title("Loan Amount Boxplot")
    for ax in axes:
        ax.set_xlabel("")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_histograms(df: pd.DataFrame, output_path: Path) -> None:
    """Plot histograms for all numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, column in zip(axes, numeric_cols):
        sns.histplot(data=df, x=column, kde=False, ax=ax, color="#8c564b", bins=30)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    """Execute the EDA pipeline and save all visualizations."""
    sns.set_style("whitegrid")
    base_path = Path(__file__).resolve().parent
    dataset_path = base_path / "dataset" / "train_u6lujuX_CVtuZ9i.csv"
    output_dir = base_path / "static" / "eda"
    create_output_directory(output_dir)

    df = load_dataset(dataset_path)
    display_dataset_overview(df)
    cleaned_df = clean_dataset(df)

    plot_countplot(
        cleaned_df,
        column="Loan_Status",
        output_path=output_dir / "loan_status_count.png",
        title="Loan Status Count Plot",
    )
    print("Loan application approvals and rejections are clearly visible in the loan status count plot.")

    plot_countplot(
        cleaned_df,
        column="Gender",
        output_path=output_dir / "gender_distribution.png",
        title="Gender Distribution",
    )
    print("Gender distribution shows applicant representation across the dataset.")

    plot_countplot(
        cleaned_df,
        column="Education",
        output_path=output_dir / "education_distribution.png",
        title="Education Distribution",
    )
    print("Education distribution reveals the balance between graduates and non-graduates.")

    plot_countplot(
        cleaned_df,
        column="Married",
        hue="Loan_Status",
        output_path=output_dir / "married_vs_loan_status.png",
        title="Married vs Loan Status",
    )
    print("Married applicants can be compared against loan status in the married vs loan status plot.")

    plot_countplot(
        cleaned_df,
        column="Self_Employed",
        hue="Loan_Status",
        output_path=output_dir / "self_employed_vs_loan_status.png",
        title="Self Employed vs Loan Status",
    )
    print("Self-employed status and loan outcomes are compared in the plot.")

    plot_countplot(
        cleaned_df,
        column="Property_Area",
        hue="Loan_Status",
        output_path=output_dir / "property_area_vs_loan_status.png",
        title="Property Area vs Loan Status",
    )
    print("Loan approval trends are visible across property area categories.")

    plot_countplot(
        cleaned_df,
        column="Credit_History",
        hue="Loan_Status",
        output_path=output_dir / "credit_history_vs_loan_status.png",
        title="Credit History vs Loan Status",
    )
    print("Applicants with Credit History = 1 have significantly higher approval chances.")

    plot_countplot(
        cleaned_df,
        column="Dependents",
        output_path=output_dir / "dependents_distribution.png",
        title="Dependents Distribution",
    )
    print("Dependents distribution helps understand household size impact.")

    plot_distribution(
        cleaned_df,
        column="ApplicantIncome",
        output_path=output_dir / "applicant_income_distribution.png",
        title="Applicant Income Distribution",
    )
    print("Applicant income distribution is visualized to evaluate applicant earning patterns.")

    plot_distribution(
        cleaned_df,
        column="CoapplicantIncome",
        output_path=output_dir / "coapplicant_income_distribution.png",
        title="Coapplicant Income Distribution",
    )
    print("Coapplicant income distribution is plotted to understand dual-income applications.")

    plot_distribution(
        cleaned_df,
        column="LoanAmount",
        output_path=output_dir / "loan_amount_distribution.png",
        title="Loan Amount Distribution",
    )
    print("Loan amount distribution highlights the range of requested financing.")

    plot_distribution(
        cleaned_df,
        column="Loan_Amount_Term",
        output_path=output_dir / "loan_amount_term_distribution.png",
        title="Loan Amount Term Distribution",
    )
    print("Loan term distribution shows the preferred repayment periods.")

    plot_correlation_heatmap(
        cleaned_df,
        output_path=output_dir / "correlation_heatmap.png",
    )
    print("Correlation heatmap reveals relationships between numerical loan features.")

    plot_pairplot(
        cleaned_df,
        output_path=output_dir / "pairplot_numerical_features.png",
    )
    print("Pairplot visualizes pairwise relationships between numeric features.")

    plot_boxplots(
        cleaned_df,
        output_path=output_dir / "income_loan_amount_boxplots.png",
    )
    print("Boxplots identify outliers in applicant income and loan amount.")

    plot_histograms(
        cleaned_df,
        output_path=output_dir / "numerical_features_histogram.png",
    )
    print("Histograms for numeric features summarize the overall distributions.")

    print("EDA completed successfully.")
    print("All visualizations saved inside static/eda")


if __name__ == "__main__":
    main()
