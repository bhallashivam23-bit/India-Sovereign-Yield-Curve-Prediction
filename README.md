# India-Sovereign-Yield-Curve-Prediction
Machine Learning Pipeline for Predicting Indian Sovereign Yield Curve Steepening and Flattening using Logistic Regression, Random Forest, XGBoost and Streamlit.
# 🇮🇳 India Sovereign Yield Curve Prediction using Machine Learning

## Project Overview

This project develops an end-to-end machine learning pipeline to analyze and predict movements in the Indian Sovereign Yield Curve. Using historical Government Security (G-Sec) yields published by the Reserve Bank of India (RBI), the project engineers fixed-income features and trains multiple machine learning models to forecast whether the yield curve is expected to **steepen** or **flatten** in the subsequent period.

The project combines financial economics, data analytics, machine learning, and dashboard development into an interactive decision-support tool.

---

## Objectives

- Construct the Indian Sovereign Yield Curve from historical G-Sec yields.
- Engineer economically meaningful fixed-income features.
- Predict future yield curve direction (Steepening vs Flattening).
- Compare traditional statistical models with ensemble learning methods.
- Build an interactive Streamlit dashboard for visualization and prediction.

---

## Dataset

- Historical Indian Government Security (G-Sec) Yield Data
- Source: Reserve Bank of India (RBI)
- Maturities Used:
  - 1Y
  - 2Y
  - 3Y
  - 5Y
  - 7Y
  - 10Y
  - 13Y

---

## Feature Engineering

The following fixed-income features were engineered:

- 2s10s Yield Spread
- 5s10s Yield Spread
- 2s10s Spread Change
- 5s10s Spread Change
- Lag-1 Spread
- Lag-3 Spread
- Rolling Volatility
- Future Spread
- Binary Target Variable (Steepening / Flattening)

---

## Machine Learning Models

The project compares three classification algorithms:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 55.38% |
| Random Forest | 58.46% |
| XGBoost | 58.46% |

The ensemble models outperformed Logistic Regression, indicating that nonlinear relationships exist within Indian yield curve dynamics.

---

## Dashboard Features

The Streamlit dashboard provides:

- Interactive Yield Curve Visualization
- Fiscal Year Selection
- Monthly Yield Curve Selection
- Yield Spread Analysis
- Machine Learning Prediction Panel
- Random Forest Prediction
- XGBoost Prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## Repository Structure

```
India-Sovereign-Yield-Curve-Prediction/

│── Yield_curve.py
│── Yield_curve_dashboard.py
│── cleaned_yield_curve_data.csv
│── random_forest_model.pkl
│── xgboost_model.pkl
│── requirements.txt
│── README.md
│── Yield_Curve_Project_Journal.pdf
```

---

## Future Improvements

- Incorporate macroeconomic indicators such as CPI Inflation, RBI Repo Rate, and GDP Growth.
- Extend forecasting to multi-class yield curve movements.
- Compare with deep learning architectures such as LSTM and Temporal Transformers.
- Deploy the application using Streamlit Community Cloud.

---

## Author

**Shivam Bhalla**

M.Sc. Financial Economics

Gokhale Institute of Politics and Economics (GIPE)
