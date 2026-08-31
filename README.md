# Churn-Prediction
Predicting which telecom customers are likely to churn, using the IBM Telco Customer Churn dataset. Built as an end-to-end classification project: data cleaning, EDA, feature engineering, model comparison, and evaluation.
## Problem

Customer acquisition costs far more than retention. This project identifies customers at high risk of churning so a business could target them with retention offers before they leave — a common use case at subscription/telecom/fintech companies.

## Dataset

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features)
- **Target:** `Churn` (Yes/No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method), and services subscribed (internet, phone, streaming, tech support, etc.)
- Included at `data/telco_churn.csv`

## Approach

1. **Cleaning** — handled missing `TotalCharges` values (blank strings for customers with 0 tenure), encoded the target and binary fields.
2. **EDA** — churn rate overall and by contract type/tenure to sanity-check assumptions before modeling.
3. **Feature engineering** — one-hot encoded categoricals, added an `AvgMonthlySpend` derived feature, standardized numeric features.
4. **Modeling** — trained and compared three classifiers: Logistic Regression, Random Forest, and Gradient Boosting, all with class balancing (churn is ~26.5% of the dataset — imbalanced).
5. **Evaluation** — compared models on accuracy, precision, recall, F1, and ROC-AUC, since accuracy alone is misleading on imbalanced data. Selected the best model by ROC-AUC.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.51 | 0.78 | 0.61 | **0.842** |
| Random Forest | 0.78 | 0.62 | 0.48 | 0.54 | 0.827 |
| Gradient Boosting | 0.80 | 0.67 | 0.51 | 0.58 | 0.842 |

**Best model: Logistic Regression** (selected for highest ROC-AUC and recall — catching more true churners matters more than raw accuracy here, since a missed churner is a lost customer).

Key churn drivers identified: month-to-month contracts, low tenure, fiber-optic internet without tech support/online security, and electronic-check payment method.

## Project Structure

```
churn-prediction/
├── data/
│   └── telco_churn.csv

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn

## Possible Extensions

- Hyperparameter tuning (GridSearchCV) for the Gradient Boosting model
- SHAP values for more interpretable feature attribution
- A simple Streamlit app to serve live predictions
- Cost-based threshold tuning (weigh false negatives vs. false positives by actual retention-offer cost)
