import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 로컬 데이터 불러오기
file_path=r".\datasets\TCS_64_04_01_705374.csv"
dataset = pd.read_csv(file_path)

# 데이터 구조 확인 (강의 종 학생들에게 보여주기 좋습니다)
print("데이터셋 정보")
print(dataset.info())
print("\n- 데이터 상위 5행 -")
print(dataset.head())

# dataset['date'] = pd.to_datetime(dataset['date'])

# 독립 변수(X: 면차)와 종속 변수〔y: 연봉) 설정
X = dataset.iloc[ :, :-1]. values
y = dataset.iloc[:, -1]. values
print(X)
print(y)
print("\n- 데이터 확인 -")

# 2. 데이터 분리 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

# 3. 모델 생성 및 학습 (내부적으로 최소자승법 OLS 실행)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 결과 예측 및 평가
y_pred = model.predict(X_test)

# 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n --- [실습 결과] --- ")
print(f"기울기(Weight, w): {model.coef_[0]:.2f}")
print(f"절편(Bias, b): {model.intercept_:.2f}")
print(f"RMSE: {rmse :.2f}")
print(f"R-squared: {r2 :.4f}")

# 5. 시각화: 모델이 찾은 '최적의 직선' 확인

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Actual Data') # OE
plt.plot( X, model.predict(X), color='royalblue', linewidth=2, label='OLS Line') # 4
plt.title('Salary vs Experience (OLS Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid( True, linestyle='-', alpha=0.6)
plt.show()