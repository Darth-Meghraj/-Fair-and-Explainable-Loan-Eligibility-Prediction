FAIR AND EXPLAINABLE LOAN ELIGIBILITY PREDICTOR

Fair and Explainable Loan Eligibility Prediction using HMDA data.
This project combines XGBoost, SHAP explainability, and fairness analysis to build a transparent, reproducible loan approval model with a Streamlit interface.

KEY FEATURES:

Data preprocessing with reproducible artifacts

Models: Logistic Regression, Random Forest, and XGBoost with Optuna tuning

SHAP explanations for global and local interpretability

Fairness analysis across demographic subgroups with bias mitigation

Robustness checks via cross-validation, error analysis, and feature ablation

Streamlit app for real-time predictions and explanations

REPOSITORY STRUCTURE:

 Streamlit app

models/ : Trained models (xgboost_best.joblib, stacking.joblib)

artifacts/ : Feature list, encodings, preprocessing logs

reports/ : SHAP plots, fairness plots, LaTeX report

notebooks/ : Jupyter notebooks for training and analysis

tests/ : Unit tests for reproducibility

requirements.txt : Python dependencies

README.md : Project overview

LICENSE : License file

Getting Started

Clone repository:
git clone https://github.com/your-username/Loan-Eligibility-ML.git

cd Loan-Eligibility-ML

Install dependencies:
pip install -r requirements.txt

Run Streamlit app:
streamlit run deploy/app.py

Usage

Enter applicant details such as loan amount, income, loan type, purpose, and property type.

Receive a loan eligibility prediction with probability.

View a SHAP-based plain-English explanation of the decision.

Try built-in example applicants (approved, declined, borderline).

Research Context

This project demonstrates how machine learning can be responsibly applied to financial decision-making by combining accuracy, transparency, and fairness. It uses the HMDA dataset as a case study and provides both a reproducible research framework and a deployable application.

License

This project is licensed under the MIT License. See LICENSE for details.
