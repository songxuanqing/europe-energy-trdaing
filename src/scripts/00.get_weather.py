from meteostat import Point, Hourly
from datetime import datetime
import asyncio
import aiofiles

# Berlin 기준
station_id = '10382' 

start = datetime(2024, 1, 1)
end = datetime(2025, 12, 31)

# Meteostat 시계열 데이터 불러오기
data = Hourly(station_id, start, end)
df_weather = data.fetch()

print(df_weather.head())

# 비동기 CSV 저장 함수
async def save_csv_async(df, filename):
    async with aiofiles.open(filename, mode='w') as f:
        # CSV로 변환 후 저장
        await f.write(df.to_csv())

# 이벤트 루프 실행
asyncio.run(save_csv_async(df_weather, 'berlin_weather_2024_2025.csv'))