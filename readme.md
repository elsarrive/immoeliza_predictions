# 🏠 Real Estate Price Prediction — Data Cleaning & Modeling

---

## Overview

This notebook presents a **data cleaning and modeling pipeline** aiming to predict real estate prices in Belgium, using enriched and carefully prepared datasets with data from Immoweb.

---

## Table of Contents

1. [Data Loading](#data-loading)
2. [Cleaning & Feature Engineering](#cleaning)
3. [Train/Test Split & Targeted Imputation](#split-impute)
4. [Preprocessing (Standardization)](#scaling)
5. [Model Training & Evaluation](#model-eval)

---

<a id="data-loading"></a>
## 1. Data Loading

- Load the main dataset of property listings (`Kangaroo.csv`).
- Enrich the data by merging with an auxiliary dataset (`data.csv`) containing latitude/longitude, energy consumption, and cadastral information.

---

<a id="cleaning"></a>
## 2. Cleaning & Feature Engineering

Using the function `cleaning_dataframe(df)`:
- **Map EPC scores (A–G)** to region-specific numerical energy values.
- **Remove or correct outliers** in various columns.
- **Advanced boolean handling** (replace NaN, convert to integer type).
- **Aggregate features** (e.g., total parking count).
- **Supervised categorical encoding** (label encoding for property subtype, province, etc.).
- **Conditional imputation and replacement** (e.g., set garden/terrace surfaces to 0 if absent).
- Drop rows with missing key features or inconsistent values.

---

<a id="split-impute"></a>
## 3. Train/Test Split & Targeted Imputation

Using the function `cleaning_traintestsplit(df_split)`:
- Applied **independently** to `X_train` and `X_test` to prevent data leakage.
- Impute missing values in key variables using mode or median, depending on data type.
- Cast selected categorical variables to `int` for modeling.
- Optionally drop rows still having missing values after imputation.

---

<a id="scaling"></a>
## 4. Preprocessing (Standardization)

- Apply standardization (`StandardScaler`) to numerical explanatory variables.
- Prepare features for the machine learning model.

---

<a id="model-eval"></a>
## 5. Model Training & Evaluation

- **Model used:** 
XGBoost Regressor, configured with:
  - `n_estimators=3000`
  - `random_state=43`
  - `learning_rate=0.05`
  - `subsample=0.8`
- **Performance metrics:**
  - **R²:** 0.83
  - **RMSE:** €77,260
  - **MAE:** €51,690
  - **MSE:** 5.97 × 10⁹

---

## Reproducibility

To use or adapt this notebook:

1. Install required Python packages: pandas, numpy, sklearn, xgboost
2. Place the data files in your working directory.
3. Run all notebook cells in order, or adapt the functions to match your own data schemas.

---

## Author / Contact

This notebook was developed by **Elsa**  
For questions or suggestions: elsaringling@gmail.com

---

## Notes

- The data pipeline can be easily customized for other real estate datasets or usage scenarios.
- Contributions and improvement suggestions are welcome! Please open an issue or pull request if using in a public repo.

---

**Happy data cleaning and modeling!** 🔎🏡📊
