import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
st.title("India Sovereign Yiels Curve Dashboard")
st.write("Fixed Income ML Dashboard")
import pandas as pd
df = pd.read_csv("cleaned_yield_curve_data.csv")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
st.dataframe(df.head(100))
st.sidebar.title("Dashboard Controls")

#### INTERACTIVE YIELD CURVE
import matplotlib.pyplot as plt
st.subheader("Interactive Yield Curve")
selected_year = st.sidebar.selectbox("Select Fiscal Year",
                             df["fiscal_year"].unique())
selected_month = st.sidebar.selectbox("Select Fiscal Month",
                             df["month_label"].unique())

# Now i'll be filtering data
filtered_df = df[
                (df["fiscal_year"] == selected_year) &
                (df["month_label"] == selected_month)
]

yield_columns = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "13Y"]

# Create plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(yield_columns, filtered_df[yield_columns].iloc[0], marker="o")
ax.set_title(f"Yield Curve - {selected_month} {selected_year}")
ax.set_xlabel("Maturity")
ax.set_ylabel("Yield %")
ax.grid(True)

# Display chart
st.pyplot(fig)

#### INTERACTIVE SPREAD
st.subheader("Yield Spread Analysis")
spread_option = st.sidebar.selectbox("Select Spread",["2s10s_spread", "5s10s_spread"])

#create time index
df["time_index"] = range(len(df))
fig2, ax2 = plt.subplots(figsize = (12,5))
ax2.plot(df["time_index"], df[spread_option])
ax2.set_title(f"{spread_option} Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Spread")
ax2.grid(True)
st.pyplot(fig2)

#ML Prediction panel
st.subheader("ML Curve Prediction")
latest_data = filtered_df.iloc[0]
feature_column = ["2s10s_spread",
                   "5s10s_spread",
                   "2s10s_change",
                   "5s10s_change",
                   "lag_1_2s10s",
                   "lag_3_2s10s",
                   "lag_1_5s10s",
                   "lag_3_5s10s",
                   "rolling_vol_2s10s",
                   "rolling_vol_5s10s"]

latest_feature = pd.DataFrame([latest_data[feature_column]])
rf_prediction = rf_model.predict(latest_feature)[0]

#Random Forest prediction
if rf_prediction == 1:
    st.success("Random Forest Model : Yield Curve Steepening")
else :
    st.error("Random Forest Model: Yield Curve Flattening")

# XGBoost Prediction
xgb_prediction = xgb_model.predict(latest_feature)[0]

if xgb_prediction == 1:
    st.success(
        "XGBoost: Yield Curve Steepening"
    )
else:
    st.error("XGBoost: Yield Curve Flattening")







