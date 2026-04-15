import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor # 튜닝이 가능한 경사하강법 모델 사용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
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

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

start_date = data['Date'].min()
data['Days'] = (data['Date'] - start_date).dt.days

X = data[['Days']]
y = data['Value']

# [STEP 1] 데이터 분할 (Train:Val:Test = 6:2:2)
# 먼저 Train과 Temp(40%)로 나눔
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
# Temp를 다시 반으로 나눠서 Validation(20%)과 Test(20%)로 분리
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# [STEP 2] 스케일링 (경사하강법 모델 사용을 위해 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# [STEP 3] 모델 학습 (Train 적용)
# 튜닝을 위해 SGDRegressor를 사용합니다.
model = SGDRegressor(max_iter=1000, eta0=0.01, random_state=42)
model.fit(X_train_scaled, y_train)

# [STEP 4] 모델 수정 및 튜닝 (Validation 적용)
# 이 단계에서 예측 결과가 좋지 않으면 max_iter나 eta0 값을 수정합니다.
y_val_pred = model.predict(X_val_scaled)
val_r2 = r2_score(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)

print(f" === [검증 단계] 중간 점검 === ")
print(f"Validation R2 Score: {val_r2:.4f}")
print(f"Validation MAE: ${val_mae:.2f}")
print(" -> 이 결과를 보고 파라미터를 수정하거나 특성을 추가합니다.\n")

# [STEP 5] 최종 성능 확인 (Test 적용)
# 모든 설정이 끝난 후 단 한 번만 수행합니다.
y_test_pred = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)

print(f" === [최종 결과] 테스트 데이터 성능 === ")
print(f"Final Test R2 Score: {test_r2:.4f}")
print(f"최종 모델은 한 번도 보지 못한 데이터에 대해 {test_r2*100:.1f}%의 설명력을 가집니다.\n")

# [STEP 6] 미래 예측 (전체 프로세스 완료 후 30일 뒤 예측)
last_day = data['Days'].max()
X_future = pd.DataFrame([[last_day + 30]], columns=['Days'])
X_future_scaled = scaler.transform(X_future)
pred_30 = model.predict(X_future_scaled)

print(f"==== 경사하강법(SGD) 분석 결과 ====")
print(f" === [예측] 30일 후 유가 === ")
print(f"예상 날짜: {(data['Date'].max() + timedelta(days=30)).date()}")
print(f"예상 유가: ${pred_30[0]:.2f}")
# 6. 모델 평가 및 미래 예측
#y_pred = model.predict(X)
X_scaled_all = scaler.transform(X) # 전체 데이터도 스케일링 적용
y_pred = model.predict(X_scaled_all)
print(f"분할 전 전체 데이터 정확도(R2): {r2_score(y, y_pred):.4f}")
# print(f"정확도(R2): {r2_score(y, y_pred):.4f}")

# 7. 시각화 (학습용과 테스트용 데이터를 구분하여 표시)
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.3, label='Train Data')
plt.scatter(X_val, y_val, color='brown', alpha=0.3, label='Verification Data')
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Test Data (Unseen)')


# 전체 추세선 (전체 구간에 대해 스케일링 후 예측)
# X_all_scaled = scaler.transform(X.sort_values(by='Days'))
# plt.plot(X.sort_values(by='Days'), model.predict(X_all_scaled), color='red', linewidth=2, label='Final Model Trend')
sorted_days = X.sort_values(by='Days')
X_all_scaled = scaler.transform(sorted_days) # 정렬된 데이터로 스케일링
plt.plot(sorted_days, model.predict(X_all_scaled), color='red', linewidth=2, label='Final Model Trend')

target_day = last_day + 30
plt.scatter(target_day, pred_30[0], color='orange', s=150, zorder=5, label='30-day Forecast')
plt.annotate(f'${pred_30[0]:.2f}', (target_day, pred_30[0]),
             textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title("Oil Price Analysis: Train/Val/Test Split Process", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()