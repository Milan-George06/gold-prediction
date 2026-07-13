# ==========================================================
# GOLD PRICE PREDICTION - COLLEGE PROJECT
# Model: Linear Regression
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import pickle


# 1. DATA COLLECTION

df = pd.read_csv("Gold Price.csv")
print("Raw data shape:", df.shape)
print(df.head())


# 2. DATA PREPROCESSING

df['Date'] = pd.to_datetime(df['Date'])

# Dataset comes newest -> oldest; sort to oldest -> newest (chronological)
df = df.sort_values('Date').reset_index(drop=True)

print("\nDuplicate rows:", df.duplicated().sum())
print("Missing values:\n", df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna()


# 3. EXPLORATORY DATA ANALYSIS (EDA)

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Price'])
plt.title("Gold Price Trend (2014 - 2026)")
plt.xlabel("Date")
plt.ylabel("Price (INR per 10g)")
plt.tight_layout()
plt.savefig("eda_price_trend.png")
plt.close()

df['MA_30'] = df['Price'].rolling(window=30).mean()
df['MA_90'] = df['Price'].rolling(window=90).mean()

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Price'], label='Price', alpha=0.5)
plt.plot(df['Date'], df['MA_30'], label='30-day MA')
plt.plot(df['Date'], df['MA_90'], label='90-day MA')
plt.title("Gold Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (INR per 10g)")
plt.legend()
plt.tight_layout()
plt.savefig("eda_moving_averages.png")
plt.close()

print("\nSummary statistics:")
print(df['Price'].describe())


# 4. FEATURE ENGINEERING

# Only yesterday's info is used to predict today's price
# (using today's own Open/High/Low/Volume would be data leakage,
# since those aren't known in advance in real life)

df['lag_1'] = df['Price'].shift(1)
df['lag_2'] = df['Price'].shift(2)
df['lag_3'] = df['Price'].shift(3)

df['ma_7']  = df['Price'].shift(1).rolling(window=7).mean()
df['ma_30'] = df['Price'].shift(1).rolling(window=30).mean()

df['prev_open']    = df['Open'].shift(1)
df['prev_high']    = df['High'].shift(1)
df['prev_low']     = df['Low'].shift(1)
df['prev_volume']  = df['Volume'].shift(1)
df['prev_chg_pct'] = df['Chg%'].shift(1)

df['target'] = df['Price']  # today's actual price

df = df.dropna().reset_index(drop=True)

feature_cols = [
    'lag_1', 'lag_2', 'lag_3',
    'ma_7', 'ma_30',
    'prev_open', 'prev_high', 'prev_low', 'prev_volume', 'prev_chg_pct'
]

X = df[feature_cols]
y = df['target']

print("\nFeatures used:", feature_cols)
print("Final dataset shape after feature engineering:", df.shape)


# 5. TRAIN-TEST SPLIT (chronological, not shuffled)

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = df['Date'].iloc[split_index:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# 6. LINEAR REGRESSION MODEL

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)


# 7. MODEL EVALUATION METRICS

mae = mean_absolute_error(y_test, preds)
rmse = sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n--- Linear Regression Results ---")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

pd.DataFrame([{"model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2": r2}]) \
  .to_csv("model_results.csv", index=False)


# 8. RESULT INTERPRETATION (predicted vs actual)

plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test.values, label="Actual Price")
plt.plot(dates_test, preds, label="Predicted Price")
plt.title("Actual vs Predicted Gold Price (Test Set) - Linear Regression")
plt.xlabel("Date")
plt.ylabel("Price (INR per 10g)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()


# SAVE TRAINED MODEL

with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved: linear_regression_model.pkl")


# 9. PREDICT NEXT DAY (no user input needed)

def predict_next_day(df, model):
    """
    Uses the most recent known row of data to predict the next day's
    gold price. Nothing is typed in by the user - all features are
    computed automatically from the existing dataset.
    """
    
    latest = df.iloc[-1]

    next_features = pd.DataFrame([{
        'lag_1': latest['Price'],
        'lag_2': df.iloc[-2]['Price'],
        'lag_3': df.iloc[-3]['Price'],
        'ma_7': df['Price'].tail(7).mean(),
        'ma_30': df['Price'].tail(30).mean(),
        'prev_open': latest['Open'],
        'prev_high': latest['High'],
        'prev_low': latest['Low'],
        'prev_volume': latest['Volume'],
        'prev_chg_pct': latest['Chg%'],
    }])

    return model.predict(next_features)[0]

next_day_pred = predict_next_day(df, model)
last_date = df['Date'].iloc[-1]
next_date = last_date + pd.Timedelta(days=1)

print(f"\nLast known date in dataset: {last_date.date()}")
print(f"Predicted price for {next_date.date()}: {next_day_pred:.2f}")

print("\nDONE. Check eda_price_trend.png, eda_moving_averages.png,")
print("predicted_vs_actual.png and model_results.csv")