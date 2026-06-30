####################### PHASE 1 LOADING DATSETS #######################
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

print()


#Loading the Yield dataset
df = pd.read_excel("C:/Users/bhall/Downloads/Month-end Yield of SGL Transactions in Government Dated Securities for Various Maturities.xlsx",
                   skiprows=4)
df.columns = df.iloc[0]
df = df[1:]
df = df.drop(columns=[df.columns[0]])
df = df[df["Term to maturity (in years)"] != "2026-27    "]
df.replace("-", np.nan, inplace=True)

####################### PHASE 2 DATA CLEANING #######################
#converting strings to numeric 
df["Term to maturity (in years)"] = pd.to_numeric(
    df["Term to maturity (in years)"],
    errors="coerce"
)
month_columns = df.columns[1:]

#CONVERT ALL MONTH COLUMNS
df[month_columns] = df[month_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Remove fake numbering row
df = df[~(
    (df["Term to maturity (in years)"] == 1.0) &
    (df["Apr."] == 2.0)
)]

# Reset index cleanly
df = df.reset_index(drop=True)

# FIX: drop garbage column-number row (1,2,3...13) and NaN spacer rows ──────
df = df.dropna(subset=["Term to maturity (in years)"])  # drop NaN maturity rows
df = df[~df["Term to maturity (in years)"].isin([        # drop year-label rows
    float(x) for x in range(1, 14)                       # that leaked as 1.0–13.0
    if df[df["Term to maturity (in years)"] == float(x)]["Apr."].eq(float(x)).any()
])]
# Simpler direct fix: drop rows where Apr. value equals Term to maturity value
# (that's the artifact row where Apr.=2.0, May=3.0 etc.)
mask = df["Apr."] == df["Term to maturity (in years)"].shift(-1).fillna(0)
df = df[df.index != df[mask].index[0]] if mask.any() else df
df = df.reset_index(drop=True)
print(df.head(15))
print(df.columns)
print(df.info())


####################### PHASE 3 VISUALISTAION AND STRUCTURING ######################## 
# #Plotting Indian Yield Curve
single_curve = df.iloc[30:60]
plt.figure(figsize=(12,6))
plt.plot(single_curve["Term to maturity (in years)"],
         single_curve["Apr."],
         marker = "o")
plt.title("Indian Sovereign Yield Curve - Apri  l")
plt.xlabel("Maturity(years)")
plt.ylabel("Yield %")
plt.grid(True)
plt.show()

# ── Build time-series: loop over all year blocks ───────────────────────────────
# Each block is 30 rows (maturities 1–30). Year labels were already dropped.
block_size = 30
n_blocks = len(df) // block_size

# Fiscal year labels in order (newest first, matching the Excel)
fiscal_years = [
    "2025-26", "2024-25", "2023-24", "2022-23", "2021-22",
    "2020-21", "2019-20", "2018-19", "2017-18", "2016-17",
    "2015-16", "2014-15", "2013-14", "2012-13", "2011-12",
    "2010-11", "2009-10", "2008-09", "2007-08", "2006-07",
    "2005-06", "2004-05", "2003-04", "2002-03", "2001-02",
    "2000-01", "1999-00", "1998-99", "1997-98", "1996-97"
]


target_tenors = [1, 2, 3, 5, 7, 10, 13]
month_order   = ["Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.",
                  "Oct.", "Nov.", "Dec.", "Jan.", "Feb.", "Mar."]

all_records = []

for i in range(min(n_blocks, len(fiscal_years))):
    block = df.iloc[i * block_size : (i + 1) * block_size].copy()
    block = block.loc[:, ~block.columns.duplicated()]           # deduplicate cols

    available = [t for t in target_tenors if t in block["Term to maturity (in years)"].values]

    # Transpose: rows = months, columns = tenors
    block_t = block.set_index("Term to maturity (in years)").T
    block_t = block_t.loc[:, ~block_t.columns.duplicated()]
    block_t = block_t[[t for t in target_tenors if t in block_t.columns]]
    block_t.columns = [f"{int(float(t))}Y" for t in block_t.columns]
    block_t.index.name = "month_label"
    block_t["fiscal_year"] = fiscal_years[i]
    all_records.append(block_t)

# Combine all years
df_all = pd.concat(all_records)
df_all = df_all.reset_index()

# Reorder months correctly within each fiscal year
df_all["month_order"] = df_all["month_label"].map(
    {m: i for i, m in enumerate(month_order)}
)
df_all = df_all.sort_values(["fiscal_year", "month_order"], ascending=[False, True])
df_all = df_all.drop(columns=["month_order"])

print(df_all.head(24))
print(df_all.shape)
print(df_all.describe())


####################### PHASE 4 FIXED INCOME FEATURES ENGINEERING #######################
###Spreads and change in spreads of different yields
#2Y-10Y Yield Spreads
df_all["2s10s_spread"] = (df_all["10Y"] - df_all["2Y"] )

#change in 2Y-10Y Yield Spreads or delta
df_all["2s10s_change"] = (df_all["2s10s_spread"].diff())


#5Y-10Y Yield Spreads
df_all["5s10s_spread"] = (df_all["10Y"] - df_all["5Y"] )

#change in 5Y-10Y Yield Spreads or delta
df_all["5s10s_change"] = (df_all["5s10s_spread"].diff())

#displaying spreads
print(df_all[[
    "fiscal_year",
    "month_label", 
    "2Y", "5Y", "10Y",
    "2s10s_spread", "2s10s_change", 
    "5s10s_spread", "5s10s_change"]].head(20))

# PLOT 2s10s SPREAD OVER TIME
plt.figure(figsize=(14,6))
plt.plot(
    df_all["2s10s_spread"],
    marker="o",
    linewidth=2
)
plt.title("Indian Yield Curve: 2s10s Spread")
plt.xlabel("Observation")
plt.ylabel("Spread (%)")
plt.grid(True)
plt.show()

# PLOT 5s10s SPREAD OVER TIME
plt.figure(figsize=(14,6))
plt.plot(
    df_all["5s10s_spread"],
    marker="o",
    linewidth=2
)
plt.title("Indian Yield Curve: 5s10s Spread")
plt.xlabel("Observation")
plt.ylabel("Spread (%)")
plt.grid(True)
plt.show()

###Lagged spreads
df_all["lag_1_2s10s"] = (df_all["2s10s_spread"].shift(1))
df_all["lag_3_2s10s"] = (df_all["2s10s_spread"].shift(3))
df_all["lag_1_5s10s"] = (df_all["5s10s_spread"].shift(1))
df_all["lag_3_5s10s"] = (df_all["5s10s_spread"].shift(3))
print(df_all[["lag_1_2s10s", "lag_3_2s10s", "lag_1_5s10s", "lag_3_5s10s"]])

###rolling volatality
df_all["rolling_vol_2s10s"] = (df_all["2s10s_spread"].rolling(window=3).std())
df_all["rolling_vol_5s10s"] = (df_all["5s10s_spread"].rolling(window=3).std())
print(df_all[["rolling_vol_2s10s", "rolling_vol_5s10s"]])

####################### PHASE 5 SUPERVISED ML TARGET ENGINEERING #######################
# Now we try to Predict whether the yield curve steepens next month or not. For eg if next month's 2s10s is greater than current the the curve steepens.
# Also this will be binary classification as the curve will either steepen next month or simply do not steepen
### Target Variable Creation
df_all["future_2s10s"] = (df_all["2s10s_spread"].shift(-1))
df_all["target_steepening_2s10s"] = (df_all["future_2s10s"] > df_all["2s10s_spread"]).astype(int) # .astype(int) converts boolean to int(true false to 1 0)
df_all["future_5s10s"] = (df_all["5s10s_spread"].shift(-1))
df_all["target_steepening_5s10s"] = (df_all["future_5s10s"] > df_all["5s10s_spread"]).astype(int)
print(df_all[["2s10s_spread",
               "future_2s10s", 
               "target_steepening_2s10s", 
               "5s10s_spread",
               "future_5s10s",
               "target_steepening_5s10s"]].head(20))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

###Building ML dataset(feature and target column)
feature_columns = ["2s10s_spread",
                   "5s10s_spread",
                   "2s10s_change",
                   "5s10s_change",
                   "lag_1_2s10s",
                   "lag_3_2s10s",
                   "lag_1_5s10s",
                   "lag_3_5s10s",
                   "rolling_vol_2s10s",
                   "rolling_vol_5s10s"]

ml_data = df_all[feature_columns + ["target_steepening_2s10s"]].dropna()
print(ml_data.head())
print(ml_data.shape)

X = ml_data[feature_columns]
Y = ml_data["target_steepening_2s10s"]
print(X.head())
print(Y.head())

###Train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

###LOGISTIC REGRESSION MODEL
model = LogisticRegression()
model.fit(X_train, Y_train)
print("Model Trainig Complete")

#prediction from LOGISTIC REGRESSION MODEL
Y_pred = model.predict(X_test)
print(Y_pred[:10])

#model evaluation
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

###RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
           n_estimators=100,
           random_state=42
)
rf_model.fit(X_train, Y_train)
print("Random Forest Complete")
import joblib
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest Saved")

rf_predict = rf_model.predict(X_test)
print(rf_predict[:10])

rf_accuracy = accuracy_score(Y_test, rf_predict)
print(rf_accuracy)
print(classification_report(Y_test,rf_predict))
print(confusion_matrix(Y_test, rf_predict))

#Feature importance 
feature_importance = pd.DataFrame({
                    "Feature" : feature_columns,
                    "Importance" : rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by="Importance", ascending= False)
print(feature_importance) #interestingly there is quite good importnace of lags which indicate presence of autocorrelation. Past curve behaviour influences future ones
#The model identified current yield curve structure and spread volatility as the strongest predictors of future curve steepening, suggesting that structural term-spread conditions and macro uncertainty dominate short-term momentum effects
#Higher spread volatility likely reflects macro uncertainty and unstable monetary expectations, which increases the probability of future curve regime changes

# #VVVVI###################Plotting feature importance 
plt.figure(figsize=(12,6))
plt.bar(feature_importance["Feature"],
        feature_importance["Importance"])
plt.xticks(rotation=45)
plt.title("Random Forest Features Importance Visualisation")
plt.ylabel("Importance Score")
plt.grid(True)
plt.show()

###XGBOOST MODEL
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
           n_estimators=100,
           max_depth = 3,
           learning_rate = 0.1,
           random_state =42
)
xgb_model.fit(X_train, Y_train)
print("XGBoost Model Completed")
joblib.dump(xgb_model, "xgboost_model.pkl")
xgb_predict = xgb_model.predict(X_test)
print("XGBoost model saved")
print(xgb_predict[:10])

#XGB Evaluation
xgb_accuracy = accuracy_score(Y_test, xgb_predict)
print(xgb_accuracy)
print(classification_report(Y_test,xgb_predict))
print(confusion_matrix(Y_test, xgb_predict))
#Although XGBoost have same accuracy as random forest, XGBoost improved recall substantially, indicating better detection of actual steepening periods, although this came with increased false positive predictions — a classic precision-recall tradeoff common in financial classification problems

df_all.to_csv("cleaned_yield_curve_data.csv", index=False)
print("Cleaned dataset saved.")










