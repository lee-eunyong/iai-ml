# [1단계] 도구 챙기기: 넘파이와 3가지 평가 지표 함수 불러오기
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# [2단계] 실험할 데이터 만들기
y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 0]) # 모델의 예측값
y_true = np.array([0, 1, 0, 0, 0, 0, 1, 1]) # 실제 정답

#
# [수동 계산을 위한 중간 점검]
# 데이터 8개를 하나씩 비교해보면 다음과 같습니다.
# TP (실제 1, 예측 1) = 2개
# FP (실제 0, 예촉 1) = 3개 (오경보)
# FN (실제 1, 예측 0) = 1개 (놓침)
# TN (실제 0, 예측 0) = 2개

# [3단계] 정밀도(Precision) 계산
# 공식: TP / (TP + FP) = 2/(2+3) = 0.4
precision = precision_score(y_true, y_pred)
print("1. 정밀도(Precision)")
print(f"결과: {precision}")
print() # 줄 띄우기

# [4단계] 민감도/재현율(Recall) 계산
# 공식: TP / (TP + FN) = 2/ (2+1)=0.666 ...
recall = recall_score(y_true, y_pred)
print("2. 민감도(Recall)")
print(f"결과: {recall}")
print()

# [5단계] F1 스코어(F1-score) 계샌
# 정밀도와 민감도의 조화평균! 둘 중 하나라도 너무 낮으면 F1 점수도 깎입니다.
f1 = f1_score(y_true, y_pred)
print("3. F1 스코어(F1-score)")
print(f"결과: {f1}")