import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 표준화
df = pd.read_csv(r".\datasets\Salary_Data.csv")
X, y = df[['YearsExperience']], df['Salary']
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# 2. 3가지 경사하강법 설정
# (1) Stochastic: 1개씩 업데이트
sgd_stochastic = SGDRegressor(max_iter=1, learning_rate='constant', eta0=0.01, random_state=42)

# (2) Mini-batch: 5개씩 묶어서 업데이트
sgd_minibatch = SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42)

# (3) Batch: 30개 전체를 보고 업데이트
sgd_batch = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01, random_state=42)

# 3. 시각화 비교
plt.figure(figsize=(18, 5))

# [2H= 1] Stochastic GD
plt.subplot( 1, 3, 1)
sgd_stochastic.partial_fit(X_s[:1], y[:1].values.ravel())
plt.scatter(X_s, y, color='orange', alpha=0.3)
plt.plot( X_s, sgd_stochastic.predict(X_s), color='red', label='Stochastic (1 sample)')
plt.title( "1. Stochastic GD", fontsize=13, color='red')
plt.legend()

# [2H= 2] Mini-batch GD
plt.subplot ( 1, 3, 2)
sgd_minibatch.partial_fit(X_s[:5], y[:5].values.ravel())
plt.scatter(X_s, y, color='orange', alpha=0.3)
plt.plot( X_s, sgd_minibatch.predict(X_s), color='blue', label='Mini-batch (5 samples)')
plt.title( "2. Mini-batch GD", fontsize=13, color='blue')
plt.legend()

# [2H= 3] Full-Batch GD
plt.subplot( 1, 3, 3)
sgd_batch.fit(X_s, y)
plt.scatter(X_s, y, color='orange', alpha=0.3)
plt.plot( X_s, sgd_batch.predict(X_s), color='green', linewidth=3, label='Batch (All 30 samples) ')
plt.title("3. Full-Batch GD", fontsize=13, color='green')
plt.legend()

plt.tight_layout()
plt.show()