import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 데이터 로드 및 전처리 (이전과 동일)
file_path = r".\dataset\iris.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['variety'])
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 모델 학습 및 예축
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# ■ 결과 표 출력 파트 ---
print("\n" + "="*50)
print(f" => 최종 모델 정확도: {accuracy_score(y_test, y_pred) * 100 :.2f}%")
print("="*50 + "\n")

# # 1: 5F 22E (Precision, Recall, F1-score)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("[표 1: 품종별 상세 분류 성능 지표]")
print(report_df.round(3)) # 소수점 3자리까지 출력
print("-" * 50 + "\n")

# 표 2: 혼동 행형 (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
labels = knn.classes_
cm_df = pd.DataFrame(cm, index=[f"실제{l}" for l in labels],
    columns=[f"예측{l}" for l in labels])

print("[표 2: 혼동 행렬 (Confusion Matrix Table)]")
print(cm_df)
print("="*50)

#