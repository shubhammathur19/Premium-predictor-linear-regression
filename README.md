# Premium-predictor-linear-regression
End-to-end linear regression study predicting medical insurance premiums (R² 0.88): assumption checks, regularization, and interaction features in Python.


What the notebook covers:----
Data import, data dictionary, stats before cleaning
EDA: target skew, feature distributions, premium by category
Correlations before and after cleaning
Outlier detection (IQR); only features are capped, never the target
Missing-value handling (mode for categorical, scaled KNN for numeric)
Hypothesis tests: Welch t-test, one-way ANOVA, chi-square
Encoding: 0/1 for binary, one-hot with `drop_first=True`
60/20/20 train/validation/test split; scaler fitted on train only
Assumption checks with PASS/WARNING verdicts: VIF, Ramsey RESET, Durbin-Watson, Breusch-Pagan, Jarque-Bera, Q-Q plot
OLS (statsmodels) for inference + scikit-learn model, USD-per-unit coefficients
Validation diagnostics, 5-fold CV via a `Pipeline` (no leakage)
Ridge, Lasso, Elastic Net with CV-tuned alpha; R² and adjusted R²
Interactions and log-target experiments
Key finding from the assumption checks: RESET, Breusch-Pagan and Jarque-Bera all flag violations (non-linearity, heteroscedasticity, non-normal residuals), which is typical for cost data and motivated the interaction features.
Data
Medical Cost Personal Datasets: 1,338 rows, no missing values. The `charges` column is renamed to `annual_premium`.
