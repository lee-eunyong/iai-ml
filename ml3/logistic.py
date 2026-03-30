import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# 1. 데이터 준비 (0, 1, 2 세 가지 숫자만 사용)
digits = datasets.load_digits()
X, y = digits.data, digits.target
mask = np.isin(y, [0, 1, 2])
X_sub, y_sub = X[mask], y[mask]

# --- [실습 1] 시그모이드 (Sigmoid): "이것은 0입니까?" --- # 0이면 1(True), 1이나 20면 0(False)으로
y_binary = (y_sub == 0).astype(int)
model_sigmoid = LogisticRegression(max_iter=10000).fit(X_sub, y_binary)
# 첫 변째 샘플에 대한 확률 ( [0이 아닐 확률, 0일 확률] )
prob_sigmoid = model_sigmoid.predict_proba(X_sub[:1])[0]

# --- [실습 2] 소프트맥스 (Softmax): "0, 1, 2 중 누구입니까?" --- |
model_softmax = LogisticRegression(max_iter=10000).fit(X_sub, y_sub)
# 첫 번째 샘플에 대한 확률 ( [0일 확률, 1일 확률, 2일 확률] )
prob_softmax = model_softmax.predict_proba(X_sub[:1])[0]

# 2. 결과 출력 및 시각화
print(f" === 확률 결과 비교 === ")
print(f"Sigmoid Output (Is it 0?): {prob_sigmoid}")
print(f"Softmax Output (0, 1, or 2?): {prob_softmax}")

plt.figure(figsize=(12, 5))

# 원쪽: 시그모이드 결과 (이진 분류 - 선택지가 2개)
# 질문: "0인가요?" -> 대달: "네(Is 0)" 혹은 "아니오(Not 0)"
plt.subplot( 1, 2, 1)
plt.bar( ['Not 0', 'Is 0'], prob_sigmoid, color=['Lightgray', 'royalblue'], alpha=0.8)
plt.title( "Sigmoid (Binary Classification)", fontsize=14)
plt.ylabel("Probability")
plt.ylim( 0, 1.1)
for i, v in enumerate(prob_sigmoid):
    plt.text(i, v + 0.02, f"{v :.2f}", ha='center')

plt.subplot( 1, 2, 2)
plt.bar( ['Digit 0', 'Digit 1', 'Digit 2'], prob_softmax, color=['tomato', 'mediumseagreen', 'orange' ], alpha=0.8)
plt.title( "Softmax (Multiclass Classification)", fontsize=14)
plt.ylabel("Probability")
plt.ylim( 0, 1.1)
for i, v in enumerate(prob_softmax):
    plt.text(i, v+0.02, f"{v :.2f}", ha='center')

plt.tight_layout()
plt.show()