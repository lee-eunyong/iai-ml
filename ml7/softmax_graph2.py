import numpy as np
import matplotlib.pyplot as plt

def softmax(values): # 1개의 사용 위치
    array_values = np.exp(values)
    return array_values / np.sum(array_values)

# --- 데이터 준비 ---
values = [-2, -1, -5, 0.5]
y = softmax(values)
labels = [f'Class {i} ({v})' for i, v in enumerate(values)]
colors = ['#4f46e5', "#818cf8", '#c7d2fe', '#f43f5e']

# --- 그래표 그리기 --·
fig, ax = plt.subplots(figsize=(12, 4))

# 단일 누적 카로 막대 그래프 생성
left_pos = 0 # 누적 시작 지점
for i in range(len(y)):
    ax.barh( 'Softmax Total (1.0)', y[i], left=left_pos, color=colors[i], label=labels[i], edgecolor='white', height=0.6)

# 막대 중앙에 확률 값(%) 표시 (비중이 너무 작으면 텍스트 생략)
    if y[i] > 0.03:
        ax.text(left_pos + y[i] / 2, 0, f'{y[i]* 100 :.1f}%',
        va='center', ha='center', color='white'if i != 2 else 'black', fontweight='bold', fontsize=11)

    left_pos += y[i]

# 그래프 꾸미기
ax.set_title( "Softmax Output: 100% Stacked Visualization", fontsize=16, pad=20)
ax.set_xlim( 0,1) # 전체 길이를 딱 1로 고정
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0 (Total)'])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

# 불필요한 테두리 제거
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()

# 검증용 출력
print("각 클래스 확를:", [f"{prob :.4f}" for prob in y])
print("확률 총합", np.sum(y))