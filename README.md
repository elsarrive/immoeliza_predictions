# 🏡 ImmoEliza: Real Estate Price Prediction Application

## Overview
ImmoEliza is a user-friendly web application that predicts property prices in Belgium using a machine learning model (XGBoost Regressor). Users can enter the characteristics of a property via a simple interface and instantly receive a price estimate.

## Technology stack:
* XGBoost for regression
* scikit-learn
* Streamlit for the web interface
* pandas, numpy

## 🚀 Getting started
1.  **Run the repository**
    ```bash
    git clone [https://github.com/yourusername/immoeliza.git](https://github.com/elsarrive/immoeliza_predictions.git)
    cd immoeliza_predictions
    ```

2.  **Installing dependencies**
    We recommend using a virtual environment (optional but encouraged):
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```
    Install dependencies :
    ```bash
    pip install -r requirements.txt
    ```
    (If you don't have any `requirements.txt`, see below for a pip-ready list).

3.  **Prepare the data**
    Place your CSV data files named `Kangaroo.csv` and `data.csv` in the project folder (it should contain the property data expected by `cleaning_dataset.py`).


## ⚙️ Prerequisites
Minimum requirements:
```
streamlit
pandas
numpy
xgboost
scikit-learn
```

## 📝 How does it work?
1.  **User input:** You fill in the web form describing your property.
2.  **Pre-processing:** The characteristics entered are formatted, transformed and cleaned according to the needs of the model.
3.  **Scaling:** The data is scaled using the same scaler as when the model was trained.
4.  **Prediction:** The trained XGBoost model produces an estimated price.
5.  **Result:** The estimated price appears instantly on the screen!

## 🧑‍💻 Developer Notes
* You can re-train or adjust the model by modifying `XGBoost_model.py`.
* Add new functionality/data - simply align the column names and data types with the pre-processing code in `cleaning_dataset.py`.
* Need to add pages or extend the user interface? Streamlit makes it easy to create multi-page applications (see documentation)!

## 📬 Contribution
Pull requests are welcome!
