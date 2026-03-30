# [1단계] 라이브러리 불러오기
from sklearn.metrics import confusion_matrix

# [2단계] 실험할 데이터 만들기
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]

# [3단계] 혼동 행렬(Confusion Matrix) 계산 및 출력
cm = confusion_matrix(y_true, y_pred)

print("1. 혼동 행렬 전체 모습")
print(cm)

# 여기에 빈 print()를 추가하면 한 줄이 깔끔하게 띄워집니다!
# (만약 두 줄을 띄우고 싶다면 print("\n") 이라고 적어주세요.)
print()

# [4단계] ravel() 함수로 2차원 행렬을 1차원으로 쫙 펴서 각각의 변수에 담기
tn, fp, fn, tp = confusion_matrix(y_true, y_pred) .ravel()

print("2. 세부 지표 값 확인")
print(f"TN (True Negative) : {tn}")
print(f"FP (False Positive): {fp}")
print(f"FN (False Negative): {fn}")
print(f"TP (True Positive) : {tp}")