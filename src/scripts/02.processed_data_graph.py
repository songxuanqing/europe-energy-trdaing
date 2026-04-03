import pandas as pd
import matplotlib
matplotlib.use("Agg")  # macOS에서 GUI 창 사용
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss



# --- 저장된 전처리 데이터 로드 ---
df = pd.read_csv("data/processed/processed_data.csv", parse_dates=["datetime"])

# --- datetime을 인덱스로 설정 ---
df.set_index("datetime", inplace=True)

# --- 예시 1: 실제 부하 vs 예측 부하 ---
plt.figure(figsize=(15,5))
plt.plot(df["actual_load"], label="Actual Load")
plt.plot(df["forecast_load"], label="Forecast Load", alpha=0.7)
plt.title("Actual vs Forecast Load")
plt.xlabel("Datetime")
plt.ylabel("Load (MWh)")
plt.legend()
plt.show()

# --- 예시 2: 풍력 + 태양광 vs 전체 부하 ---
plt.figure(figsize=(15,5))
plt.plot(df["actual_load"], label="Actual Load")
plt.plot(df["wind_generation"] + df["solar"], label="Wind + Solar Generation", alpha=0.7)
plt.title("Renewable Generation vs Actual Load")
plt.xlabel("Datetime")
plt.ylabel("MWh")
plt.legend()
plt.show()

# --- 예시 3: 재생에너지 비율 ---
plt.figure(figsize=(15,5))
plt.plot(df["renewable_ratio"], label="Renewable Ratio")
plt.title("Renewable Energy Ratio")
plt.xlabel("Datetime")
plt.ylabel("Ratio")
plt.legend()
plt.show()

# 예: 24시간 단위 이동평균
df['actual_load_ma24'] = df['actual_load'].rolling(window=24, center=False).mean()

# 예: 7일 단위 이동평균 (일별 데이터 기준)
df['actual_load_ma7d'] = df['actual_load'].rolling(window=7, center=False).mean()

# 이동평균(롤링 평균) 적용
plt.figure(figsize=(15,5))
plt.plot(df['actual_load'], label='Original Load', alpha=0.5)
plt.plot(df['actual_load_ma24'], label='24h Moving Average', color='red')
plt.title('Load with 24h Moving Average')
plt.xlabel('Datetime')
plt.ylabel('Load (MWh)')
plt.legend()
plt.show()

#이동 표준편차(롤링 변동) 확인
df['rolling_std24'] = df['actual_load'].rolling(window=24).std()

plt.figure(figsize=(15,5))
plt.plot(df['actual_load'], label='Original Load', alpha=0.5)
plt.plot(df['rolling_std24'], label='24h Rolling Std', color='green')
plt.title('Rolling Standard Deviation')
plt.xlabel('Datetime')
plt.ylabel('Load (MWh)')
plt.legend()
plt.show()



# Augmented Dickey-Fuller (ADF) 테스트
result = adfuller(df['actual_load'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
    print('Critical Value (%s): %.3f' % (key, value))


# 차분(differencing)으로 정상화
df['actual_load_diff'] = df['actual_load'] - df['actual_load'].shift(1)

plt.figure(figsize=(15,5))
plt.plot(df['actual_load_diff'], label='Differenced Load', color='orange')
plt.title('Differenced Load for Stationarity')
plt.xlabel('Datetime')
plt.ylabel('Load Difference (MWh)')
plt.legend()
plt.show()

# 원 시계열
result = adfuller(df['actual_load'].dropna())
print('ADF Statistic (original):', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# 차분 후 시계열
result_diff = adfuller(df['actual_load_diff'].dropna())
print('ADF Statistic (1st differenced):', result_diff[0])
print('p-value:', result_diff[1])
print('Critical Values:', result_diff[4])

# 시계열 분해 (추세, 계절성, 잔차) 가법 분해
decomposition = seasonal_decompose(df['actual_load'], model='additive', period=24)
decomposition.plot()
plt.show()

# KPSS test
stat, p_value, lags, crit_values = kpss(df['actual_load'].dropna(), regression='c', nlags="auto")
print('KPSS Statistic:', stat)
print('p-value:', p_value)
print('Critical Values:', crit_values)