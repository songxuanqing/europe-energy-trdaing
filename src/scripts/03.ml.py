import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from lightgbm import log_evaluation

# -------------------------------
# 1️⃣ 데이터 로드
# -------------------------------
df = pd.read_csv(
    "data/processed/processed_data.csv",
    parse_dates=["datetime"]
)

# 정렬 (시계열 필수)
df = df.sort_values("datetime")

# -------------------------------
# 2️⃣ 결측치 처리
# -------------------------------
df = df.ffill().dropna()

# -------------------------------
# 3️⃣ Target 정의
# -------------------------------
if "forecast_error" not in df.columns:
    df["forecast_error"] = df["actual_load"] - df["forecast_load"]

# -------------------------------
# 4️⃣ Feature 선택 (업데이트)
# -------------------------------
features = [
    # 시간 정보
    "hour", "weekday", "month", "is_weekend", "is_holiday",

    # 수요/예측
    "actual_load", "forecast_load",

    # 발전 / 기상
    "temperature", "dew_point", "humidity", "precipitation", "snow_depth",
    "wind_direction", "wind_speed", "wind_gust", "pressure", "weather_code",
    "wind_generation", "solar", "conventional_generation",

    # 전력 흐름
    "net_import_export",

    # lag (핵심🔥)
    "load_lag_1", "load_lag_24", "load_lag_48", "load_lag_168",
    "forecast_error_lag_1", "forecast_error_lag_24",
    "forecast_error_lag_48", "forecast_error_lag_168",

    # rolling
    "load_roll_24", "load_roll_168", "error_roll_24",

    # 파생 feature
    "wind_solar_total", "renewable_ratio"
]

target = "forecast_error"

X = df[features]
y = df[target]

# -------------------------------
# 5️⃣ Train/Test Split (시계열)
# -------------------------------
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("df shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# 6️⃣ 모델 정의
# -------------------------------
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# -------------------------------
# 7️⃣ 학습
# -------------------------------
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l1",
    callbacks=[log_evaluation(100)]
)

# -------------------------------
# 8️⃣ 예측
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9️⃣ 성능 평가
# -------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mae, rmse, mape

mae, rmse, mape = evaluate(y_test, y_pred)

print("\n===== Forecast Error Model Performance =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

# -------------------------------
# 1️⃣0️⃣ 결과 분석
# -------------------------------
df_test = df.iloc[split:].copy()
df_test["predicted_error"] = y_pred

# Alpha Signal
df_test["alpha_signal"] = df_test["predicted_error"]
df_test["pred_direction"] = (df_test["predicted_error"] > 0).astype(int)
df_test["real_direction"] = (df_test["forecast_error"] > 0).astype(int)
direction_acc = (df_test["pred_direction"] == df_test["real_direction"]).mean()
print(f"\nDirection Accuracy: {direction_acc:.4f}")

# 큰 오차 탐지
threshold = df["forecast_error"].std() * 2
df_test["big_error"] = (np.abs(df_test["forecast_error"]) > threshold).astype(int)
df_test["pred_big_error"] = (np.abs(df_test["predicted_error"]) > threshold).astype(int)
big_error_acc = (df_test["big_error"] == df_test["pred_big_error"]).mean()
print(f"Big Error Detection Accuracy: {big_error_acc:.4f}")

# Feature Importance
import matplotlib.pyplot as plt
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Features:")
print(importance.head(15))

plt.figure()
plt.barh(importance["feature"][:15], importance["importance"][:15])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.show()

# 결과 저장
df_test.to_csv("data/processed/forecast_error_predictions.csv", index=False)