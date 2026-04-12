# 서해안(A)과 경부선(B) 지점 간의 시차 상관관계 확인 예시
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import ccf


def load_HIWY_data():
    base_path = Path(__file__).parent
    csv_path = base_path / "datasets" / "EX_SPOT_HIWY_WETHER_HISTR.csv"
    return pd.read_csv(csv_path)
# 데이터 로드 및 시계열 설정

df = load_HIWY_data()

west_coast = df[df['WTCD_RGDVS_CD'] == '140']['VSBT_DIST']
gyeongbu = df[df['WTCD_RGDVS_CD'] == '133']['VSBT_DIST']

# 시차 상관계수 계산
correlation = ccf(west_coast, gyeongbu)
# 상관계수가 가장 높은 시차(lag)가 안개 이동에 걸리는 예상 시간입니다.