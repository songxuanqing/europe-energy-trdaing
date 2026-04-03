# model.py
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature import preprocess  # 전처리 후 df 가져오기

def train_model(csv_path=None, df=None):
    if csv_path is not None:
        # CSV 경로가 주어지면 읽어오기
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df.set_index("datetime", inplace=True)
    else:
        # 경로도 없으면 preprocess() 호출
        df = preprocess()

    # Feature / Target
    features = [
        "hour","weekday","month","is_weekend","is_holiday",
        "actual_load","forecast_load",
        "temperature","dew_point","humidity","precipitation","snow_depth",
        "wind_direction","wind_speed","wind_gust","pressure","weather_code",
        "wind_generation","solar","conventional_generation",
        "net_import_export",
        "load_lag_1","load_lag_24","load_lag_48","load_lag_168",
        "forecast_error_lag_1","forecast_error_lag_24",
        "forecast_error_lag_48","forecast_error_lag_168",
        "load_roll_24","load_roll_168","error_roll_24",
        "wind_solar_total","renewable_ratio"
    ]
    target = "forecast_error"

    X = df[features]
    y = df[target]

    # Train/Test split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 모델 정의
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # 학습
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              eval_metric="l1", callbacks=[log_evaluation(100)])

    # 예측
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, features, df.iloc[split:].copy()

# 평가 함수
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mae, rmse, mape

if __name__ == "__main__":
    model, X_test, y_test, y_pred, features, df_test = train_model()
    mae, rmse, mape = evaluate(y_test, y_pred)
    print("\n===== Forecast Error Model Performance =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")