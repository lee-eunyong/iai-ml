import numpy as np

def softmax(values): #1개의 사용 위치

    # 1. 각 요소에 지수 함수(e^x)를 적용합니다.
    array_values = np.exp(values)

    # 2. 각 요소를 전체 합으로 나누어 결과의 총합이 1이 되도록 정규화합니다.
    sumval = np.sum(array_values)
    print(f"0. 입력값에 지수함수 적용한 것의 합계 {sumval}")
    print(f"0. 입력값 {array_values[0]}")
    print(f"1. 입력값 {array_values[1]}")
    print(f"2. 입력값 {array_values[2]}")
    print(f"3. 입력값 {array_values[3]}")
    return array_values / sumval # np.sum(array_values)

values = [-2, -1, -5, 0.5]

# 소프트맥스 함수 실행
y = softmax(values)

print("1. 입력값 (Values):", values)
print("2. 소프트맥스 결과 (y):")

for i, prob in enumerate(y):
    print(f" - 클래스 {i}의 확률: {prob:.8f} ({prob * 100 :.2f}%)")

# 결과의 합이 1인지 확인
total_sum = y.sum()
print("\n3. 결과값의 총합 (y.sum()):", total_sum)

# 결과 해석
max_index = np.argmax(y)
print(f"\n 가장 높은 확률을 가진 인덱스는 {max_index}번이며, 값은 {values[max_index]}입니다.")