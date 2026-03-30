import pandas as pd
from pathlib import Path

def load_housing_data():
    base_path = Path(__file__).parent
    csv_path = base_path / "datasets" / "housing.csv"
    return pd.read_csv(csv_path)

if  __name__ == "__main__":
    # Load data
    housing = load_housing_data()

    # Step 1: Initial check
    print("\nStep 1: Checking for missing values")
    print(housing.isnull().sum())

    # Step 2: Processing
    print("\nStep 2: Selecting Missing Data Strategy")

    # [Option 1] Drop rows with missing values
    # housing = housing.dropna(subset=["total_bedrooms"])
    # print("Result: Dropped rows containing missing values.")

    # [Option 2] Drop the entire column
    # housing = housing.drop("total_bedrooms", axis=1)
    # print("Result: Dropped the 'total_bedrooms' column.")

    # [Option 3] Impute with median (Recommended)
    median = housing["total_bedrooms"] .median()
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
    print(f"Result: Imputed missing values with median ({median}).")

    # Step 3: Final verification
    print("\nstep 3: Verification after processing")
    print(housing.isnull().sum())



