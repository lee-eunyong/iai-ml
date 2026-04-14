import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import SGDRegressor  # 경사하강법 회귀 모델
from sklearn.preprocessing import StandardScaler # 스케일링 도구
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

start_date = data['Date'].min()
data['Days'] = (data['Date'] - start_date).dt.days

X = data[['Days']]
y = data['Value']

# 2. 데이터 스케일링 (경사하강법 필수 단계)
# Days(0~29000일)와 Value의 단위 차이가 크면 경사하강법이 발산하거나 느려집니다.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 모델 생성 및 학습 (Stochastic Gradient Descent)
# max_iter: 최대 반복 횟수, eta0: 학습률(Learning Rate)
model_sgd = SGDRegressor(max_iter=1000, eta0=0.01, random_state=42)
model_sgd.fit(X_scaled, y)

# 4. 학습 결과 및 평가
y_pred = model_sgd.predict(X_scaled)

# 5. 30일 후 미래 예측
last_date = data['Date'].max()
target_date = last_date + timedelta(days=30)
target_days = (target_date - start_date).days

# 미래 데이터도 반드시 기존 학습된 scaler로 변환해야 함
# X_new_scaled = scaler.transform([[target_days]])
# 학습할 때 사용한 컬럼명 'Days'를 명시한 데이터프레임으로 변환
X_new = pd.DataFrame([[target_days]], columns=['Days'])
X_new_scaled = scaler.transform(X_new)
pred_30 = model_sgd.predict(X_new_scaled)

print(f" === 경사하강법(SGD) 분석 결과 === ")
print(f"1) 분석 기준 마지막 날짜: {last_date.date()}")
print(f"2) 30일 후 예측 날짜: {target_date.date()}")
print(f"3) 예상 유가: ${pred_30[0]:.2f}")
print(f"4) 모델 정확도(R2): {r2_score(y, y_pred):.4f}")
print(f"5) 평균 절대 오차(MAE): ${mean_absolute_error(y, y_pred):.2f}")

# 6. 시각화
plt.figure(figsize=(12, 6))
plt.scatter(data['Date'], y, color='blue', alpha=0.4, label='Actual Data')
plt.plot(data['Date'], y_pred, color='orange', linewidth=2, label='SGD Regression Line')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.scatter(target_date, pred_30, color='green', s=100, zorder=5, label='Future Prediction')
plt.annotate(f'${pred_30[0]:.2f}', (target_date, pred_30[0]),
             textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title("Oil Value Analysis (Gradient Descent)", fontsize=15)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()