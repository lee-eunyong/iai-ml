import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 파일 경로 수정
df = pd.read_excel("./drowsy_driving_sample.xlsx")

print(df.head())

corr = df[["pull-offs(km)", "rest area(km)"]].corr()

print(corr)

plt.title("istst")
plt.show()



# 거리 계산 함수 (Haversine 공식)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# 쉼터 데이터 예시 (위도/경도)
rest_areas = pd.DataFrame({
    "위도": [36.1, 36.5, 36.8],
    "경도": [127.2, 127.5, 127.8]
})

def get_min_distance(row, rest_df):
    distances = rest_df.apply(
        lambda x: haversine(row['위도'], row['경도'], x['위도'], x['경도']),
        axis=1
    )
    return distances.min()

# 적용
df["계산된쉼터거리"] = df.apply(lambda row: get_min_distance(row, rest_areas), axis=1)

print(df.head())


# 쉼터 데이터 예시 (위도/경도)
rest_areas = pd.DataFrame({
    "위도": [36.1, 36.5, 36.8],
    "경도": [127.2, 127.5, 127.8]
})

def get_min_distance(row, rest_df):
    distances = rest_df.apply(
        lambda x: haversine(row['위도'], row['경도'], x['위도'], x['경도']),
        axis=1
    )
    return distances.min()

# 적용
df["계산된쉼터거리"] = df.apply(lambda row: get_min_distance(row, rest_areas), axis=1)

print(df.head())
