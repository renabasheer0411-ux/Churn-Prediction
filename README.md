# Churn-Prediction
Predicting which telecom customers are likely to churn, using the IBM Telco Customer Churn dataset. Built as an end-to-end classification project: data cleaning, EDA, feature engineering, model comparison, and evaluation.
## Problem

Customer acquisition costs far more than retention. This project identifies customers at high risk of churning so a business could target them with retention offers before they leave — a common use case at subscription/telecom/fintech companies.

## Dataset

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features)
- **Target:** `Churn` (Yes/No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method), and services subscribed (internet, phone, streaming, tech support, etc.)
- Included at `data/telco_churn.csv`

## Project Structure

├── Churn Prediction.ipynb        
├── telco_churn.csv 

## Approach

### 1. Data Cleaning
- `TotalCharges` arrives as a string column and contains 11 blank entries — these turn out to be customers with `tenure == 0` (brand-new accounts that haven't been billed yet), not random noise. Coerced to numeric with `pd.to_numeric(errors="coerce")` and imputed with the column median rather than dropping the rows, since 11 rows is small but every row of a minority-class-heavy dataset is worth keeping.
- Dropped `customerID` — a unique identifier carries no predictive signal and risks the model latching onto it as a spurious feature.
- Encoded the target (`Churn`: Yes/No → 1/0) and `SeniorCitizen` to a consistent integer type early, before any downstream split, so train/test encoding never drifts apart.

### 2. Exploratory Data Analysis
Before touching any model, the EDA step exists to sanity-check assumptions and catch anything that would silently break the pipeline:
- **Class balance** — churn sits at ~26.5% of customers. This single number drives several later decisions: using `class_weight="balanced"` in the linear/tree models, and picking ROC-AUC/recall over raw accuracy as the metrics that matter.
- **Churn by contract type** — month-to-month customers churn at a dramatically higher rate than one- or two-year contracts, which matches the business intuition that lock-in reduces churn and confirms `Contract` is a genuinely useful feature, not noise.
- **Tenure distribution by churn status** — churned customers cluster heavily at low tenure (leave early or not at all), which flagged `tenure` as one of the strongest signals well before any model confirmed it via feature importance.
- Both plots are saved to `outputs/eda_overview.png` so the reasoning is visible, not just asserted.

### 3. Feature Engineering
- Binary Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) mapped to 0/1 rather than one-hot encoded, since one-hot on a true binary just doubles columns for no benefit.
- Remaining categoricals (`Contract`, `PaymentMethod`, `InternetService`, the six service add-ons, etc.) one-hot encoded with `drop_first=True` to avoid the dummy-variable trap.
- Added one derived feature, `AvgMonthlySpend = TotalCharges / tenure`, to capture spending intensity independent of how long someone's been a customer — two customers with the same `TotalCharges` but very different tenures are in very different situations, and the raw columns alone don't express that.
- Standardized all numeric features with `StandardScaler` — necessary for Logistic Regression to converge properly and to keep coefficients comparable; harmless for the tree-based models.
- Split 80/20 with `stratify=y` so the ~26.5% churn rate is preserved in both train and test sets — an unstratified split risks a test set that under- or over-represents churners purely by chance.


### 4. Modeling
Trained three classifiers spanning different modeling assumptions, all with `class_weight="balanced"` (or default balancing where applicable) since the 3:1 class imbalance would otherwise bias every model toward predicting "no churn":
- **Logistic Regression** — a strong, interpretable baseline; coefficients are directly readable as churn drivers, which matters when the eventual audience is a retention team, not just a leaderboard.
- **Random Forest** — captures non-linear interactions (e.g., fiber-optic internet *combined with* no tech support) that logistic regression can't.
- **Gradient Boosting** — typically squeezes out extra performance by correcting prior trees' errors sequentially; included as the higher-capacity comparison point.

### 5. Evaluation
- Reported accuracy, precision, recall, F1, and ROC-AUC for every model — accuracy alone is misleading here, since a model that always predicts "no churn" would still score ~74% accuracy while being useless.
- Prioritized **ROC-AUC and recall** for model selection: in a retention use case, missing an actual churner (false negative) costs a lost customer, while flagging a loyal customer for a retention offer (false positive) mostly just costs a discount coupon. That asymmetry is why Logistic Regression's higher recall (0.78) won out over Gradient Boosting's higher precision (0.67) despite both tying on ROC-AUC.
- Generated a confusion matrix and ROC curve for the selected model, plus a feature-importance plot (for the tree-based models) to make the "why" behind predictions inspectable rather than a black box.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.51 | 0.78 | 0.61 | **0.842** |
| Random Forest | 0.78 | 0.62 | 0.48 | 0.54 | 0.827 |
| Gradient Boosting | 0.80 | 0.67 | 0.51 | 0.58 | 0.842 |

**Best model: Logistic Regression** (selected for highest ROC-AUC and recall — catching more true churners matters more than raw accuracy here, since a missed churner is a lost customer).

Key churn drivers identified: month-to-month contracts, low tenure, fiber-optic internet without tech support/online security, and electronic-check payment method.



## Usage

1. **Run the Notebook**:
   - Open `Churn Prediction.ipynb` in Google Colab.
   - Ensure that the dataset is available on Google Drive or within the local environment.
2. **Explore Outputs**
3. **Reuse the Model**
   

## Technologies Used

* **Python:** Data cleaning, feature engineering, and model building with Pandas and scikit-learn.
* **Jupyter/Google Colab:** Environment for running the pipeline and generating visualizations.
* **Power BI / Tableau:** For creating interactive dashboards (future step).
* **Libraries:** Matplotlib, Seaborn for data visualization; scikit-learn for modeling and evaluation; joblib for model persistence.

No deep learning framework or cloud infra is used — this is a classical ML pipeline deliberately kept to a stack that's fast to run, easy to read, and standard for tabular business data.


## Future Enhancements

* **Predictive Modeling:** Apply hyperparameter tuning and SHAP-based interpretability to sharpen churn predictions and explain individual results.
* **Advanced Visualization:** Build dynamic dashboards in Tableau or Power BI for real-time churn monitoring.
* **Deep Dive into Customer-Segment Analysis:** Explore churn patterns at the individual customer-segment level rather than only in aggregate.
* **Deployment:** Wrap the trained model in a simple Streamlit or Flask app to serve live churn predictions.
* **Cost-Based Threshold Tuning:** Move off the default 0.5 classification threshold and optimize it against real retention-offer cost vs. customer lifetime value.

