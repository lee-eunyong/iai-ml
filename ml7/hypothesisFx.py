import numpy as np

# [1단계] 이전에 만든 시그모이드 함수
def sigmoid(z): # 1개의 사용 위치
        return 1 / (1 + np.exp(-z))

# [2단계] 가설 함수 (Hypothesis Function) 정의
# 공식: h(x) = sigmoid(theta^T * x)
def hypothesis_function(x, theta): # 1개의 사용 위치
    # np.dot은 행렬의 곱(내적)을 계산해주는 함수입니다.
    # z = theta[0]*x[0] + theta[1]*x[1] ... 와 같은 계산을 한 줄로 끝냅니다.
    z = np.dot(x, theta)
    return sigmoid(z)

# [3단계] 실제 데이터로 테스트
# 예: 온도(x1)=30도, 진동(x2)=0.8 일 때
x_data = np.array([30, 0.8])
theta_data = np.array([0.1, 5.0])

# 최종 확률 계산
prob = hypothesis_function(x_data, theta_data)

print(f"종합 점수(z): {np.dot(x_data, theta_data)}")
print(f"최종 고장 확률: {prob * 100 :.2f}%")