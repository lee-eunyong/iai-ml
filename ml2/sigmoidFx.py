import numpy as np
import matplotlib.pyplot as plt

# [1단계] 시그모이드 함수 정의 만들기
def sigmoid(z): #4개의 사용 위치
        return 1 / (1 + np.exp(-z))

print("[PART 1] 시그모이드 함수 수치 테스트 ")

# 1. 일력값이 딱 0일 때 (정학히 절반의 확를!)
s_0 = sigmoid(0)
print(f"입력값 0 => 압축된 확률: {s_0:.5f} (50%)")

# 2. 아주 큰 양수일 때 (100% 확신에 수)
s_100 = sigmoid(100)
print(f"입력값 => 압축된 확률: {s_100 :.5f} (약 100%)")

# 3. 아주 작은 음수일 때 (0% 확신에 수련)
s_m100 = sigmoid(-100)
print(f"입력값 -100 => 압축된 확률: {s_m100:.5f} (약 0%)")

# [PART 2] 뱃플롯립(matpLotlib)으로 시그모이드 곡선 시각화하기
# 1. 그래프를 그릴 X축 데이터(날것의 점수 z) 만들기
z_values = np.linspace(-10, 10, 200)

# 2. 만든 ×축 테이터를 시그모이드 터널에 통과시키기 (Y축 확률값 획득!)
probabilities = sigmoid(z_values)

# 3. 그래프 크기 설청
plt.figure(figsize=(10, 6))

# 4. 시그모이드 S자 곡선 그리키 (발간색 긁은 선)
plt.plot( z_values, probabilities, color='red', linewidth=3, label='Sigmoid Curve')

# Y축 기준선 (상한선 1.0, 하한선 0.0)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

# 50% 합격 기준선 (임계값 0.5)
plt.axhline(y=0.5, color='black', linestyle=':', label='Threshold(0.5)')

# X축 기준선 (입력간 0)
plt.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)

# 아까 콘솔에서 테스트한 (0, 0.5) 지점에 파란 점 찍기
plt.scatter( 0, 0.5, color='blue', s=100, zorder=5, label='sigmoid(0)=0.5')

# 5. 그래프 꾸미기 (제목, 축 이름, 격자 무늬, 범례)
plt.title('Sigmoid Function (Magic Compressor)')
plt.xlabel('Raw Score (z) - from AI model') # X축: AI 모델의 '날것 점수
plt.ylabel('Probability (0.0 ~ 1.0) - output') # Y축: 압축된 확률값
plt.grid( True, linestyle='-', alpha=0.3) # 배경 격자
plt.legend(loc='upper left') # 법례를 왼쪽 위에 표시

# 6. 파이참에서 완성된 그래프 창 희우기
plt.show()