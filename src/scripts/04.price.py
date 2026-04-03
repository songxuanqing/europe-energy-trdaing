
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ 데이터 로드
# -------------------------------
df = pd.read_csv("data/processed/processed_data.csv", parse_dates=["datetime"])

# 이미 forecast_error 계산되어 있다고 가정
# df["forecast_error"] = df["actual_load"] - df["forecast_load"]

# -------------------------------
# 2️⃣ 큰 오차 / 가격 spike 정의
# -------------------------------
# forecast error spike: ±2*std 기준
error_threshold = df["forecast_error"].std() * 2
df["error_spike"] = (np.abs(df["forecast_error"]) > error_threshold).astype(int)

# 가격 spike: ±2*std 기준
price_threshold = df["day_ahead_price"].std() * 2
df["price_spike"] = (np.abs(df["day_ahead_price"] - df["day_ahead_price"].mean()) > price_threshold).astype(int)

print(f"Forecast error spike rate: {df['error_spike'].mean():.2%}")
print(f"Price spike rate: {df['price_spike'].mean():.2%}")

# -------------------------------
# 3️⃣ 상관관계 분석
# -------------------------------
# 시각화: error spike vs price spike
plt.figure(figsize=(6,6))
sns.heatmap(pd.crosstab(df['error_spike'], df['price_spike']), annot=True, fmt="d", cmap="Blues")
plt.title("Forecast Error Spike vs Price Spike")
plt.xlabel("Price Spike")
plt.ylabel("Forecast Error Spike")
plt.show()

# 숫자로 확인 (accuracy)
spike_corr = (df['error_spike'] & df['price_spike']).sum() / df['error_spike'].sum()
print(f"Error spike leading to price spike: {spike_corr:.2%}")

# -------------------------------
# 4️⃣ alpha signal 생성
# -------------------------------
# 단순 예: forecast error spike가 발생하면 alpha=1, 아니면 0
df["alpha_signal"] = df["error_spike"]

# 방향성까지 활용 (오차가 양수면 long, 음수면 short)
df["alpha_direction"] = df["forecast_error"].apply(lambda x: 1 if x > 0 else -1)

# 실제 적용 예: alpha_signal * alpha_direction
df["alpha_position"] = df["alpha_signal"] * df["alpha_direction"]

# -------------------------------
# 5️⃣ 시각화 예제
# -------------------------------
plt.figure(figsize=(15,4))
plt.plot(df["datetime"], df["forecast_error"], label="Forecast Error")
plt.plot(df["datetime"], df["day_ahead_price"] - df["day_ahead_price"].mean(), label="Price deviation")
plt.scatter(df.loc[df["error_spike"]==1, "datetime"], 
            df.loc[df["error_spike"]==1, "forecast_error"], 
            color="red", label="Error Spike", s=50)
plt.title("Forecast Error vs Price Deviation with Spike Highlight")
plt.xlabel("Datetime")
plt.ylabel("MW / EUR deviation")
plt.legend()
plt.show()

# -------------------------------
# 6️⃣ 결과 저장
# -------------------------------
df.to_csv("data/processed/forecast_error_price_analysis.csv", index=False)
print("✅ Analysis CSV saved: data/processed/forecast_error_price_analysis.csv")