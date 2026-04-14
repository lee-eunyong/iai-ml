import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # 날짜 포맷 설정을 위한 모듈
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

# 4. 학습 결과 및 평가
y_pred = model.predict(X)
w1 = model.coef_[0]
w0 = model.intercept_

# 5. 30일 후 미래 예측
last_date = data['Date'].max()
target_date = last_date + timedelta(days=30)  # 마지막 데이터로부터 정확히 30일 후
target_days = (target_date - start_date).days

X_new = pd.DataFrame([[target_days]], columns=['Days'])
pred_30 = model.predict(X_new)

print(f" === AI 분석 결과 === ")
print(f"1) 분석 기준 마지막 날짜: {last_date.date()}")
print(f"2) 30일 후 예측 날짜: {target_date.date()}")
print(f"3) 예상 유가: ${pred_30[0]:.2f}")
print(f"4) 모델 정확도(R2): {r2_score(y, y_pred):.4f}")

# 6. 시각화 (X축 날짜 포맷 수정)
plt.figure(figsize=(12, 6))
plt.scatter(data['Date'], y, color='blue', alpha=0.4, label='Actual Data')
plt.plot(data['Date'], y_pred, color='red', linewidth=2, label='Regression Line')

# --- X축 눈금 설정 (년-월 단위) ---
ax = plt.gca() # 현재 축 가져오기
# 1년 단위로 큰 눈금 표시 (더 세밀하게 하려면 mdates.MonthLocator(interval=6) 등으로 수정 가능)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
# 날짜 포맷을 'YYYY-MM' 형태로 설정
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 최신 데이터 시점 강조 (30일 후 예측 포인트)
plt.scatter(target_date, pred_30, color='green', s=100, zorder=5, label='Future Prediction (30d)')
plt.annotate(f'${pred_30[0]:.2f}', (target_date, pred_30[0]),
             textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title("Oil Value Analysis & 30-Day Forecast", fontsize=15)
plt.xlabel("Date (Year-Month)")
plt.ylabel("Price ($)")
plt.xticks(rotation=45) # 날짜가 겹치지 않게 회전
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()