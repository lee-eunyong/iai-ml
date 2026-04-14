import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
df = pd.read_csv(r'.\datasets\Salary_Data.csv')
X, y = df[ ['YearsExperience' ]], df['Salary']

# 2. OLS (수학적 정답)
ols = LinearRegression(). fit(X, y)

# 3. SGD 학습 준비 (표준화 필수)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# [중간용] 딱 1번만 업데이트한 모델
sgd_short = SGDRegressor(max_iter=1, tol=None, eta0=0.01, random_state=42).fit(X_s, y)
# [오른쪽용] 충분히 학습한 모델
sgd_long = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, random_state=42).fit(X_s, y)

# 4. 그래프 출력
plt.figure(figsize=(18, 6))

# [원쪽] OLS: 수학적 정답
plt.subplot( 1, 3, 1)
plt.scatter(X, y, color='orange', alpha=0.5)
plt.plot( X, ols.predict(X), color='blue', linewidth=3)
title_ols = f"OLS: Exact Solution\nw: {ols.coef_[0]: .2f}, b: {ols.intercept_:.2f}"
plt.title(title_ols, fontsize=12, fontweight='bold')

# [중간] SGD (1회 학습): 방황하는 단계
plt.subplot( 1, 3, 2)
plt.scatter(X_s, y, color='orange', alpha=0.5)
plt.plot( X_s, sgd_short. predict(X_s), color='red', linestyle='--', label='1 Iteration')
title_short = f"SGD: 1 Iteration\nw: {sgd_short.coef_[0] :.2f}, b: {sgd_short.intercept_[0]:.2f}"
plt.title(title_short, fontsize=12, color='red')
plt.legend()

# [오른쪽] SGD (완료〕: 수렴한 단계
plt.subplot( 1, 3, 3)
plt.scatter(X_s, y, color='orange', alpha=0.5)
plt.plot( X_s, sgd_long.predict(X_s), color='green', linewidth=3, label='1000 Iterations')
title_long = f"SGD: Final Convergence\nw: {sgd_long.coef_[0] :.2f}, b: {sgd_long.intercept_[0]:.2f}"
plt.title(title_long, fontsize=12, color='green', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()