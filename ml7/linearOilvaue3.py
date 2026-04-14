import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta

# 1. 데이터 불러오기 (경로 단순화)
file_path = r".\datasets\oil_value.csv"
data = pd.read_csv(file_path)

# [Option 1] Drop rows with missing values
data = data.dropna(subset=["Value"])
print("Result1: Dropped rows containing missing values.")

# [Option 1-1] Drop rows with missing values, 삭제
data = data.dropna(subset=["Date"])
print("Result2: Dropped rows containing missing values.")

# 2. 데이터 전처리
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

# 기준일로부터 경과된 일수 계산
start_date = data['Date'].min()
data['Days'] = (data['Date'] - start_date).dt.days

X = data[['Days']]
y = data['Value']

# 3. 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 4. 1년 후(365일) 미래 예측
last_date = data['Date'].max()
future_date = last_date + timedelta(days=365)
future_days = (future_date - start_date).days

X_future = pd.DataFrame([[future_days]], columns=['Days'])
pred_future = model.predict(X_future)
y_pred = model.predict(X)

# --- 핵심 수정 부분 ---
# pred_future는 배열 형태이므로 [0]을 붙여 단일 숫자로 추출합니다.
predicted_value = pred_future[0]

print(f" === AI 1년 후 예측 결과 === ")
print(f"1) 데이터 마지막 날짜: {last_date.date()}")
print(f"2) 1년 후 예측 날짜: {future_date.date()}")
print(f"3) 1년 후 예상 유가: ${predicted_value:.2f}")
print(f"4) 모델 정확도(R2): {r2_score(y, y_pred):.4f}")

# 5. 시각화
plt.figure(figsize=(12, 7))
plt.scatter(data['Date'], y, color='blue', alpha=0.4, label='Actual Data')

# 회귀선 그리기 (미래까지 연장)
all_dates = pd.date_range(start=start_date, end=future_date, freq='D')
all_days = (all_dates - start_date).days.values.reshape(-1, 1)
all_pred = model.predict(pd.DataFrame(all_days, columns=['Days']))
plt.plot(all_dates, all_pred, color='red', linewidth=2, label='Regression Trend')

# --- 핵심 수정 부분 (좌표에 predicted_value 사용) ---
plt.scatter(future_date, predicted_value, color='green', s=120, zorder=5, label='1-Year Forecast')
plt.annotate(f'${predicted_value:.2f}', (future_date, predicted_value),
             textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# X축 눈금 설정
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.title("Oil Value Analysis & 1-Year Future Forecast", fontsize=15)
plt.xlabel("Date (Year-Month)")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.1)
plt.tight_layout()
plt.show()