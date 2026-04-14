import pandas as pd
import numpy as np
from pathlib import Path
from dataScalingNormalization import scNor_HIWY_data
from splitdata import split_train_test_data
from statsmodels.tsa.stattools import ccf

base_path = Path(__file__).parent

def load_HIWY_data():
    csv_path = base_path / "datasets" / "oil_value.csv"
    return pd.read_csv(csv_path, encoding="UTF-8")

if  __name__ == "__main__":
    # Load data
    hiwy = load_HIWY_data()

    # base_path = Path(__file__).parent
    # output_dir = base_path / "datasets"
    # split_train_test_data(hiwy, output_dir)

    # Step 1: Initial check
    print("\nStep 1: Checking for missing values")
    print(hiwy.isnull().sum())

    # Step 2: Processing
    print("\nStep 2: Selecting Missing Data Strategy")

    # [Option 1] Drop rows with missing values
    hiwy = hiwy.dropna(subset=["Value"])
    print("Result1: Dropped rows containing missing values.")

    # [Option 1-1] Drop rows with missing values, 삭제
    hiwy = hiwy.dropna(subset=["Date"])
    print("Result2: Dropped rows containing missing values.")

    # [Option 2] Drop the entire column
    # hiwy = hiwy.drop("total_bedrooms", axis=1)
    # print("Result: Dropped the 'total_bedrooms' column.")

    # [Option 3] Impute with median (Recommended)
    median = hiwy["Value"].median()
    hiwy["Value"] = hiwy["Value"].fillna(median)
    print(f"Result3: Imputed missing values with median ({median}).")

    # -99를 NaN으로 변환, 결측치로 값이 없음으로 표기
    #hiwy['VSBT_DIST'] = hiwy['VSBT_DIST'].replace(-99, np.nan)
    #print("Result4: Imputed NaN containing missing values.")
    # 변환 후 결측치 개수 확인
    print(hiwy.isnull().sum())

    # hiwy = hiwy.dropna(subset=["WTCD_RGDVS"])
    #hiwy = hiwy.dropna(subset=["VSBT_DIST"])
    #print("Result4: Dropped rows containing missing values.")

    # Step 3: Final verification
    print("\nstep 3: Verification after processing")
    print(hiwy.isnull().sum())

    #hiwy = scNor_HIWY_data(hiwy)

    print("Result4: Final Data.")
    print("-" * 50)
    print(hiwy)
    print("-" * 50)

    output_dir = base_path / "datasets"
    split_train_test_data(hiwy, output_dir)

    # 5. Step 6: Min/Max verification
    # print("\nStep 6: Min/Max values after Normalization")
    # print("-" * 50)
    # print(f"Min value (All features): {hiwy.min()}")
    # print(f"Max value (All features): {hiwy.max()}")

    # west_coast = hiwy[hiwy['WTCD_RGDVS_CD'] == '140']['VSBT_DIST']
    # gyeongbu = hiwy[hiwy['WTCD_RGDVS_CD'] == '133']['VSBT_DIST']

    # 시차 상관계수 계산
    # correlation = ccf(west_coast, gyeongbu)
    # 상관계수가 가장 높은 시차(lag)가 안개 이동에 걸리는 예상 시간입니다.
    # print(correlation)