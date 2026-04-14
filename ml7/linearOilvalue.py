import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 데이터 불러오기 (경로 단순화)
file_path = r".\datasets\oil_value.csv"
data = pd.read_csv(file_path)

# [Option 1] Drop rows with missing values
data = data.dropna(subset=["Value"])
print("Result1: Dropped rows containing missing values.")

# [Option 1-1] Drop rows with missing values, 삭제
data = data.dropna(subset=["Date"])
print("Result2: Dropped rows containing missing values.")

# 2. 데이터 전처리 (날짜를 숫자로 변환)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

# 기준일(가장 오래된 날짜)로부터 경과된 '일수(Days)'를 계산하여 숫자로 변환
start_date = data['Date'].min()
data['Days'] = (data['Date'] - start_date).dt.days

# X는 수치화된 'Days', y는 'Value'
X = data[['Days']]
y = data['Value']

# 3. 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 4. 학습 결과 확인
w1 = model.coef_[0]   # 하루당 유가 인상액
w0 = model.intercept_ # 1946년(기준일) 당시의 추정 유가

print(f"==== 최소제곱법 (OLS)을 사용하여 예측 분석 결과 ====")
print(f"공식: Value = {w1:.4f} * Days + {w0:.2f}")
print(f"1) 하루당 평균 인상액: ${w1:.4f}")
print(f"2) 데이터 시작 시점 추정가: ${w0:.2f}")

# 5. 모델 평가 및 미래 예측
y_pred = model.predict(X)
print(f"3) 정확도(R2): {r2_score(y, y_pred):.4f}")

# 마지막 데이터 시점으로부터 30일 후 예측
last_day = data['Days'].max()
X_new = pd.DataFrame([[last_day + 30]], columns=['Days'])
pred_30 = model.predict(X_new)
print(f"4) 마지막 데이터 기준 30일 후 예상 유가: ${pred_30[0]:.2f}")

# 6. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(data['Date'], y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Oil Value Analysis (1946-2026)", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
