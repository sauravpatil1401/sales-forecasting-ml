import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("sales_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# 2. Data cleaning
# -----------------------------
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Drop rows with missing date/sales
df = df.dropna(subset=["date", "sales"])

# -----------------------------
# 3. Create time-based features
# -----------------------------
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Lag features
df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)

# Rolling mean
df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()

# Remove rows created with NaN lag values
df = df.dropna().reset_index(drop=True)

# -----------------------------
# 4. Basic EDA
# -----------------------------
print("\nDataset shape:", df.shape)
print("\nStatistics:\n", df["sales"].describe())

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["sales"])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("sales_trend.png")
plt.close()

# -----------------------------
# 5. Time-based train/test split
# -----------------------------
features = [
    "day",
    "day_of_week",
    "month",
    "week_of_year",
    "lag_1",
    "lag_7",
    "rolling_7",
]

X = df[features]
y = df["sales"]

# Keep the last 20% for testing
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -----------------------------
# 6. Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# 8. Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# -----------------------------
# 9. Actual vs Predicted
# -----------------------------
results = pd.DataFrame({
    "date": df.loc[X_test.index, "date"],
    "actual_sales": y_test.values,
    "predicted_sales": predictions.round(2),
})

results.to_csv("forecast_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(results["date"], results["actual_sales"], label="Actual")
plt.plot(results["date"], results["predicted_sales"], label="Predicted")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nResults saved to forecast_results.csv")
print("Charts saved as sales_trend.png and actual_vs_predicted.png")
