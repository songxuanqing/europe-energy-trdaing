# src/process.py
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

# --- 파일 경로 ---
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- CSV 파일 로드 함수 ---
def load_csv(file_name, sep=";", thousands=","):
    path = INPUT_DIR / file_name
    df = pd.read_csv(path, sep=sep, thousands=thousands)
    return df

# --- 데이터 전처리 ---
def preprocess():
    # 1. Actual consumption
    actual_df = load_csv("Actual_consumption_2024_2025_Hour.csv")
    actual_df.rename(columns={
        "Start date": "datetime",
        "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": "actual_load"
    }, inplace=True)
    actual_df["datetime"] = pd.to_datetime(actual_df["datetime"], format="%b %d, %Y %I:%M %p")
    actual_df = actual_df[["datetime", "actual_load"]]

    # 2. Forecasted consumption
    forecast_df = load_csv("Forecasted_consumption_2024_2025_Hour.csv")
    forecast_df.rename(columns={
        "Start date": "datetime",
        "grid load [MWh] Calculated resolutions": "forecast_load"
    }, inplace=True)
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], format="%b %d, %Y %I:%M %p")
    forecast_df = forecast_df[["datetime", "forecast_load"]]

    # 3. Actual generation
    generation_df = load_csv("Actual_generation_2024_2025_Hour.csv")
    generation_df.rename(columns={
        "Start date": "datetime",
        "Wind offshore [MWh] Calculated resolutions": "wind_offshore",
        "Wind onshore [MWh] Calculated resolutions": "wind_onshore",
        "Photovoltaics [MWh] Calculated resolutions": "solar",
        "Lignite [MWh] Calculated resolutions": "lignite",
        "Hard coal [MWh] Calculated resolutions": "hard_coal",
        "Fossil gas [MWh] Calculated resolutions": "gas",
        "Nuclear [MWh] Calculated resolutions": "nuclear",
        "Hydro pumped storage [MWh] Calculated resolutions": "hydro_pumped"
    }, inplace=True)
    generation_df["datetime"] = pd.to_datetime(generation_df["datetime"], format="%b %d, %Y %I:%M %p")

    # 숫자형 전환 및 NaN 처리
    conv_cols = ["lignite","hard_coal","gas","nuclear","hydro_pumped"]
    generation_df[conv_cols] = generation_df[conv_cols].astype(str) \
        .replace(",", "", regex=True) \
        .replace("-", np.nan)
    generation_df[conv_cols] = generation_df[conv_cols].apply(pd.to_numeric, errors="coerce")
    generation_df[conv_cols] = generation_df[conv_cols].interpolate(method="linear").fillna(0)

    # 합계 계산
    generation_df["conventional_generation"] = generation_df[conv_cols].sum(axis=1)
    generation_df["wind_generation"] = generation_df["wind_offshore"] + generation_df["wind_onshore"]
    generation_df = generation_df[["datetime","wind_generation","solar","conventional_generation"]]

    # 4. Cross-border flows
    crossborder_df = load_csv("Cross-border_physical_flows_2024_2025_Hour.csv")
    crossborder_df.rename(columns={
        "Start date": "datetime",
        "Net export [MWh] Calculated resolutions": "net_import_export"
    }, inplace=True)
    crossborder_df["datetime"] = pd.to_datetime(crossborder_df["datetime"], format="%b %d, %Y %I:%M %p")
    crossborder_df = crossborder_df[["datetime","net_import_export"]]

    # 5. Day-ahead prices
    price_df = load_csv("Day-ahead_prices_2024_2025_Hour.csv")
    price_df.rename(columns={
        "Start date": "datetime",
        "Germany/Luxembourg [€/MWh] Calculated resolutions": "day_ahead_price"
    }, inplace=True)
    price_df["datetime"] = pd.to_datetime(price_df["datetime"], format="%b %d, %Y %I:%M %p")
    price_df = price_df[["datetime","day_ahead_price"]]

    # 6. Weather
    weather_df = pd.read_csv(INPUT_DIR / "berlin_weather_2024_2025.csv")

    # tsun 제외하고 필요한 컬럼만 이름 변경
    weather_df.rename(columns={
        "time": "datetime",
        "temp": "temperature",
        "dwpt": "dew_point",
        "rhum": "humidity",
        "prcp": "precipitation",
        "snow": "snow_depth",
        "wdir": "wind_direction",
        "wspd": "wind_speed",
        "wpgt": "wind_gust",
        "pres": "pressure",
        "coco": "weather_code"
    }, inplace=True)

    # 사용하고자 하는 컬럼만 선택, tsun 제거
    weather_df = weather_df[[
        "datetime","temperature","dew_point","humidity","precipitation",
        "snow_depth","wind_direction","wind_speed","wind_gust","pressure","weather_code"
    ]]

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], format="%Y-%m-%d %H:%M:%S")

    # --- 데이터 병합 ---
    df = actual_df.merge(forecast_df, on="datetime", how="left")
    df = df.merge(generation_df, on="datetime", how="left")
    df = df.merge(crossborder_df, on="datetime", how="left")
    df = df.merge(price_df, on="datetime", how="left")
    df = df.merge(weather_df, on="datetime", how="left")

    # --- Numeric 컬럼 결측치 안전 처리 ---
    # merge 후, lag/rolling 만들기 전에
    numeric_cols = [
        "actual_load","forecast_load","wind_generation","solar",
        "conventional_generation","net_import_export",
        "day_ahead_price","temperature","dew_point","humidity",
        "precipitation","snow_depth","wind_direction","wind_speed",
        "wind_gust","pressure"
    ]
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # datetime을 index로 설정
    df.set_index("datetime", inplace=True)

    # 시간 기준 선형 보간
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")

    # 남은 NaN은 뒤쪽 값으로 채움
    df[numeric_cols] = df[numeric_cols].fillna(method="bfill")

    # 다시 datetime을 컬럼으로
    df.reset_index(inplace=True)

    # --- Forecast error ---
    df["forecast_error"] = df["actual_load"] - df["forecast_load"]

    # --- 시간 관련 feature ---
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # --- 공휴일 ---
    de_holidays = holidays.DE()
    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in de_holidays)

    # --- lag features ---
    for lag in [1,24,48,168]:
        df[f"load_lag_{lag}"] = df["actual_load"].shift(lag)
        df[f"forecast_error_lag_{lag}"] = df["forecast_error"].shift(lag)

    # --- rolling statistics ---
    df["load_roll_24"] = df["actual_load"].rolling(24).mean()
    df["load_roll_168"] = df["actual_load"].rolling(168).mean()
    df["error_roll_24"] = df["forecast_error"].rolling(24).mean()

    # --- renewable ratio ---
    df["wind_solar_total"] = df["wind_generation"] + df["solar"]
    df["renewable_ratio"] = df["wind_solar_total"] / df["actual_load"]

    # --- 결측치 처리: lag/rolling feature로 인한 초기 NaN 제거 ---
    lag_rolling_cols = [f"load_lag_{lag}" for lag in [1,24,48,168]] + \
                       [f"forecast_error_lag_{lag}" for lag in [1,24,48,168]] + \
                       ["load_roll_24","load_roll_168","error_roll_24"]
    df.dropna(subset=lag_rolling_cols, inplace=True)

    # --- 최종 파일 저장 ---
    df.to_csv(OUTPUT_DIR / "processed_data.csv", index=False)
    print("Preprocessed data saved to:", OUTPUT_DIR / "processed_data.csv")


if __name__ == "__main__":
    preprocess()