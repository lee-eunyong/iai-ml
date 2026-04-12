import numpy as np

# [1단계] 기본 함수 셋팅 (시그모이드, 가설, 비용)
def sigmoid(z): # 1개의 사용 위치
    return 1 / (1 + np.exp(-z))
def hypothesis_function(x, theta): # 2개의 사용 위치
    z = np.dot(x, theta)
    return sigmoid(z)
def compute_cost(x, y, theta):# 2개의 사용 위치
    m = y.shape[0]
    h = hypothesis_function(x, theta)
    # 안전한 로그 계산을 위해 아주 작은 값(1e-5)을 더해줍니다.
    term1 = y.T.dot(np.log(h +1e-5))
    term2= (1 - y).T.dot(np.log(1-h+1e-5))
    return (-1.0 / m) * (term1 + term2)

# [2단계] 경사하강법 함수 (가중치 업데이트)
def minimize_gradient(x, y, theta, iterations=1000, alpha=0.01): # 1개의 사용 위치
    m = y.size
    cost_history = []
    for _ in range(iterations):
        # 예촉값과 실제값의 차이(에러) 계산
        h = hypothesis_function(x, theta)
        loss = h - y
        # 기울기(Gradient) 계산: 에러와 데이터의 곱
        # x. T.dot(loss)는 모든 데이터에 대해 에러*데이터를 합산해좁니다.
        gradient = x.T.dot(loss) / m
        # 가중치 업데이트 (공식: theta = theta - 보폭 * 기울기)
        theta = theta - (alpha * gradient)
        # 100번마다 현재 비용(반성 점수) 기록
        if (_ % 100) == 0:
            current_cost = compute_cost(x, y, theta)
            cost_history.append(current_cost)
            print(f"반복 횟수 { _: >4} : 현재 비용(Cost) = {current_cost :.5f}")
        return theta, cost_history

# [3단계] 실제 데이터 대입 및 결과 확인 # 테스트용 데이터 (문제 3개, 특징 2개)
X_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test = np.array([1, 0, 1])
initial_theta = np.array([0.0, 0.0]) # 초기 가중치 (아무것도 모르는 상태: 0, 0)

print("9 학습을 시작합니다 ... ")# 학습 실챙 (보폭 0.01로 1000번 반복)
final_theta, history = minimize_gradient(X_test, y_test, initial_theta, iterations=1000, alpha=0.01)
print("☑ 학습 완료!")
print(f"최종 가중치(theta): {final_theta}")
print(f"최종 비용(Cost): {compute_cost(X_test, y_test, final_theta) :.5f}")