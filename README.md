# Telco Customer Churn Prediction

An end-to-end machine learning pipeline predicting customer churn for a telecom provider, built inside a Docker container for reproducibility. Covers exploratory analysis, data cleaning, feature encoding, model comparison, hyperparameter tuning, and evaluation.

The headline result is not the accuracy figure. It is that **the model with 80.5% accuracy was missing 44% of the customers who actually churned** — and what it took to find and fix that.

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features, binary target.

---

## Pipeline

```
EDA  ──▶  Preprocessing  ──▶  Model training  ──▶  Tuning  ──▶  Evaluation
 │             │                    │               │              │
 data          encoding,       LogReg, RF,      GridSearchCV   confusion matrix,
 quality,      stratified      XGBoost          on F1          threshold tuning,
 imbalance     split, scaling                                  feature importance
```

Environment: Python 3.11 in Docker, JupyterLab, pandas · scikit-learn · XGBoost · matplotlib · seaborn.

---

## Findings

### 1. A numeric column hiding as text, and 11 invisible nulls

`TotalCharges` loaded as a string rather than a float. `df.info()` reported **zero missing values** across all columns — because the blanks were stored as whitespace, which pandas counts as valid strings.

Converting with `pd.to_numeric(errors='coerce')` exposed **11 NaNs**. Investigating them showed every one belonged to a customer with `tenure = 0`: brand-new customers who had not yet been billed. The missingness was not random — it carried meaning.

Imputed **0** rather than dropping the rows or using the mean. Zero months billed means zero charged; the mean would have invented ~$1,400 of billing history for customers who had none, and dropping them would have systematically removed the newest segment.

### 2. Class imbalance makes accuracy misleading

The target splits **73.5% retained / 26.5% churned**. A model predicting "no churn" for every customer scores **73.5% accuracy** while identifying zero churners — useless to a business whose entire goal is intervening before customers leave.

That baseline became the reference point for every subsequent result, and the reason evaluation focused on recall and F1 rather than accuracy.

### 3. The simpler model beat the ensembles

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Logistic Regression | 80.5% | **80.6%** | ~0 pts |
| Random Forest | 99.8% | 79.0% | **21 pts** |
| XGBoost | 94.2% | 77.2% | **17 pts** |

Both tree ensembles ran with default hyperparameters, grew unrestricted, and memorised the training data. Logistic Regression — with no tuning at all — generalised better than either.

### 4. Tuning fixed the overfitting but did not change the winner

A 5-fold cross-validated grid search over 24 XGBoost configurations, optimising **F1 rather than accuracy**, selected `max_depth=3` — the shallowest option offered, confirming that depth was the cause.

| XGBoost | Train | Test | Gap |
|---|---|---|---|
| Untuned | 94.2% | 77.2% | 17 pts |
| Tuned | 82.0% | 80.1% | **2 pts** |

Training accuracy *dropped* by 12 points and test accuracy *rose* by 3. That is regularisation working as intended. But the tuned model only drew level with Logistic Regression — it never overtook it.

### 5. What accuracy was hiding

With two models tied at ~80%, accuracy could no longer distinguish them. The confusion matrix could.

Of **374 actual churners** in the test set:

| Model | Caught | Missed | Churn recall |
|---|---|---|---|
| Logistic Regression | 208 | **166** | 0.56 |
| Tuned XGBoost | 195 | 179 | 0.52 |

An 80.6% accuracy model was **failing to detect 44% of departing customers**. The score looked respectable because the model is excellent at the easy majority class — 90% recall on customers who stay, against 56% on those who leave.

Logistic Regression won on the metric that matters: it catches 13 more churners at identical precision.

### 6. Threshold tuning: the largest single improvement

Classifiers output probabilities; the 0.5 cutoff converting them to labels is an arbitrary default, not a business decision. Sweeping it:

| Threshold | Precision | Recall | F1 | Churners caught |
|---|---|---|---|---|
| 0.50 | 0.658 | 0.556 | 0.603 | 208 |
| 0.40 | 0.571 | 0.668 | 0.616 | 250 |
| 0.35 | 0.545 | 0.703 | 0.614 | 263 |
| **0.30** | **0.521** | **0.757** | **0.617** | **283** |
| 0.25 | 0.498 | 0.807 | 0.616 | 302 |

Lowering the threshold to **0.30** raised recall from 56% to 76% — **75 additional at-risk customers identified** — while F1 *improved* rather than degrading. The default was simply wrong for this problem.

Precision falls to 0.52, so roughly half of flagged customers will not actually churn. For churn that trade is correct: a missed customer costs their lifetime value, a false positive costs one retention offer.

### 7. Feature importance confirmed the EDA

The strongest coefficients matched what the exploratory plots suggested before any model was trained:

**Toward retention:** `Contract_Two year` (−1.32), `tenure` (−1.25), `Contract_One year` (−0.69)
**Toward churn:** `InternetService_Fiber optic` (+1.05), `PaymentMethod_Electronic check` (+0.39)

Month-to-month customers churned at ~43% versus under 3% on two-year contracts. Fiber optic emerged as the single largest churn driver — a business signal worth investigating on pricing or service quality.

`TotalCharges` (+0.52) and `MonthlyCharges` (−0.30) carry opposing signs despite the EDA showing churners paying more monthly. This is **multicollinearity**: `TotalCharges ≈ tenure × MonthlyCharges`, so the three features share the signal and individual coefficients are not independently interpretable.

---

## Final model

**Logistic Regression, decision threshold 0.30**

| Metric | Value |
|---|---|
| Churn recall | 0.757 |
| Churn precision | 0.521 |
| F1 (churn) | 0.617 |
| Churners caught | 283 of 374 |

Chosen over tuned XGBoost on recall, and additionally for being faster, simpler, and interpretable — its coefficients translate directly into business recommendations.

---

## Reproducing

```bash
git clone https://github.com/Fahad00888/telco-churn-prediction.git
cd telco-churn-prediction
docker build -t churn-env .
docker run --rm -p 8888:8888 -v "$(pwd)":/app churn-env
```

Open `http://localhost:8888` and run `01_eda.ipynb`. The dataset downloads from source on first run; the notebook saves a local copy so subsequent runs work offline.

---

## Limitations

- **Single train/test split for final evaluation.** Cross-validation was used during tuning, but the reported test metrics come from one 20% holdout. Repeated splits would give confidence intervals rather than point estimates.
- **The 0.30 threshold is not cost-optimised.** It was selected on F1 and a qualitative argument about relative error costs. With real figures for customer lifetime value and retention offer cost, the optimal threshold should be derived, not chosen.
- **No class-weighting or resampling compared.** `class_weight='balanced'` and SMOTE are the obvious alternatives to threshold tuning and were not benchmarked against it.
- **`gender` retained as a feature.** Kept for completeness, but demographic attributes in a deployed retention model warrant scrutiny — the model should not learn to treat customers differently on that basis.
- **Static dataset.** No temporal validation. Real churn behaviour drifts, and a production model would need monitoring and periodic retraining.
- **Not deployed.** The model is trained and evaluated but not served behind an API or interface.

## Future work

- Cost-sensitive threshold selection using actual CLV and retention-offer figures
- Benchmark `class_weight='balanced'` and SMOTE against threshold tuning
- SHAP values for per-customer explanations rather than global coefficients
- Serve the model behind a small API or Streamlit interface

---

## Stack

Python 3.11 · pandas · scikit-learn · XGBoost · matplotlib · seaborn · Docker · JupyterLab

**Author:** Fahad Khalid — [github.com/Fahad00888](https://github.com/Fahad00888)
