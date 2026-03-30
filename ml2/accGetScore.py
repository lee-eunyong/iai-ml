# [1단계] 필요한 라이브러리 불러오기
# 배열 계산을 위한 numpy와 정확도 계산을 위한 accuracy_score를 가져옵니다.
import numpy as np
from sklearn.metrics import accuracy_score

# [2단계] 실험할 데이터 만들기 (넘파이 배열 형태로 변환)
y_pred = np.array([0, 1, 1, 0]) # 모델이 예측한 값
y_true = np.array([0, 1,0, 0]) # 실제 정답

# [3단계] 원리 이해하기: 넘파이(NumPy)를 이용해 '직접' 정확도 계산하기
manual_accuracy = sum(y_true == y_pred) / len(y_true)

print(" 1. 직접 계산한 정확도 ")
print(f"맞춘 개수(3) / 전체 개수(4) = {manual_accuracy}")
print( )

# [4단계] 실무 활용하기: 사이킷런(scikit-Learn) 함수로 '한 번에' 계산하기
sklearn_accuracy = accuracy_score(y_true, y_pred)

print(" 2. 사이킷런 함수(accuracy_score)를 사용한 정확도 ")
print(f"결과: {sklearn_accuracy}")