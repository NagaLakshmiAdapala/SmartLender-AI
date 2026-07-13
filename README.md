# Smart Lender Loan Eligibility Prediction System

A complete end-to-end loan eligibility prediction system built with Python, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and XGBoost.

## Project Structure

- `app.py` - Flask web application for loan eligibility prediction.
- `train_model.py` - Data loading, preprocessing, model training, evaluation, and saving.
- `preprocessing.py` - Feature engineering, encoding, scaling, and transformation utilities.
- `requirements.txt` - Python package dependencies.
- `dataset/` - Contains data files used to train and evaluate the model.
- `models/` - Saved machine learning artifacts and model files.
- `templates/` - HTML templates for the Flask app.
- `static/` - CSS, JS, and image assets.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your loan dataset CSV in `dataset/loan_data.csv`.
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run the Flask app:
   ```bash
   python app.py
   ```
6. Open the browser at `http://127.0.0.1:5000`.

## Features

- Data preprocessing with encoding and scaling.
- XGBoost classifier for loan approval prediction.
- Model evaluation and visualization.
- Web UI for making predictions.

## Notes

- Use realistic loan application data and align feature names with the sample dataset.
- Add new features or custom business logic as needed.
