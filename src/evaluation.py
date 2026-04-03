# evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_from_csv(csv_path="data/processed/processed_data.csv",
                     result_path="data/processed/forecast_error_price_analysis.csv",
                     datetime_col="datetime",
                     threshold_factor=2.0):
    """
    외부 CSV를 읽어서 forecast error / price spike 분석, alpha signal 생성, 시각화 후 저장

    Parameters
    ----------
    csv_path : str or Path
        분석할 CSV 파일 경로 (forecast_error, day_ahead_price 컬럼 필요)
    result_path : str or Path
        결과 CSV 저장 경로
    datetime_col : str
        datetime 컬럼 이름
    threshold_factor : float
        spike 정의 시 표준편차 배수
    """
    df = pd.read_csv(csv_path, parse_dates=[datetime_col])

    # -------------------------------
    # 1️⃣ 큰 오차 / 가격 spike 정의
    # -------------------------------
    error_threshold = df["forecast_error"].std() * threshold_factor
    df["error_spike"] = (np.abs(df["forecast_error"]) > error_threshold).astype(int)

    price_threshold = df["day_ahead_price"].std() * threshold_factor
    df["price_spike"] = (np.abs(df["day_ahead_price"] - df["day_ahead_price"].mean()) > price_threshold).astype(int)

    print(f"Forecast error spike rate: {df['error_spike'].mean():.2%}")
    print(f"Price spike rate: {df['price_spike'].mean():.2%}")

    # -------------------------------
    # 2️⃣ 상관관계 분석
    # -------------------------------
    plt.figure(figsize=(6,6))
    sns.heatmap(pd.crosstab(df['error_spike'], df['price_spike']), annot=True, fmt="d", cmap="Blues")
    plt.title("Forecast Error Spike vs Price Spike")
    plt.xlabel("Price Spike")
    plt.ylabel("Forecast Error Spike")
    plt.show()

    spike_corr = (df['error_spike'] & df['price_spike']).sum() / df['error_spike'].sum()
    print(f"Error spike leading to price spike: {spike_corr:.2%}")

    # -------------------------------
    # 3️⃣ alpha signal 생성
    # -------------------------------
    df["alpha_signal"] = df["error_spike"]
    df["alpha_direction"] = df["forecast_error"].apply(lambda x: 1 if x > 0 else -1)
    df["alpha_position"] = df["alpha_signal"] * df["alpha_direction"]

    # -------------------------------
    # 4️⃣ 시각화 예제
    # -------------------------------
    plt.figure(figsize=(15,4))
    plt.plot(df[datetime_col], df["forecast_error"], label="Forecast Error")
    plt.plot(df[datetime_col], df["day_ahead_price"] - df["day_ahead_price"].mean(), label="Price deviation")
    plt.scatter(df.loc[df["error_spike"]==1, datetime_col],
                df.loc[df["error_spike"]==1, "forecast_error"],
                color="red", label="Error Spike", s=50)
    plt.title("Forecast Error vs Price Deviation with Spike Highlight")
    plt.xlabel("Datetime")
    plt.ylabel("MW / EUR deviation")
    plt.legend()
    plt.show()

    # -------------------------------
    # 5️⃣ 결과 저장
    # -------------------------------
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)  # 디렉토리 없으면 생성
    df.to_csv(result_path, index=False)
    print(f"✅ Analysis CSV saved: {result_path}")