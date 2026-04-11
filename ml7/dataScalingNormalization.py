import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scNor_HIWY_data(hiwy):

    # 2. Preprocessing: Handle Missing Values (Required before scaling)
    # median = hiwy["VSBT_DIST" ].median()
    # hiwy["VSBT_DIST"] = hiwy[ "PRSNT_WETHER_CN" ].fillna(median)

    # 3. Step 4: Check original statistics before scaling
    print("\nStep 4: Check original statistics before scaling")
    print("-" * 50)
    # Select only numeric features (Drop 'PRSNT_WETHER_CN' for scaling)
    hiwy = hiwy.drop("PRSNT_WETHER_CN", axis=1)  ## drop string
    print("Result5: Dropped the 'PRSNT_WETHER_CN' column.")
    print(hiwy.describe())

    hiwy = hiwy.drop("WTCD_RGDVS", axis=1)  ## drop string
    print("Result6: Dropped the 'WTCD_RGDVS' column.")
    print(hiwy.describe())

    # 4. Step 5: Scaling & Normalization
    std_scaler = StandardScaler() # Standardization
    min_max_scaler = MinMaxScaler() # Normalization

    # fit_transform returns a Numpy array
    print("\nStep 6: Standardization & Normalization")
    hiwy_std = std_scaler.fit_transform(hiwy)
    hiwy_minmax = min_max_scaler.fit_transform(hiwy)

    print("\nStep 7: Compare results of Scaling (VSBT_DIST column)")
    print("-" * 50)
    # median_income is at index 7
    print(f"Original value:      {hiwy['VSBT_DIST'].iloc[0]}")
    print(f"Standardized result: {hiwy_std[0, 3]:.4f}")
    print(f"Normalized result :  {hiwy_minmax[0, 3]:.4f}")

    # 5. Step 6: Min/Max verification
    print("\nStep 8: Min/Max values after Normalization")
    print("-" * 50)
    print(f"Min value (All features): {hiwy_minmax.min()}")
    print(f"Max value (All features): {hiwy_minmax.max()}")

    return hiwy_minmax