from sklearn.model_selection import train_test_split

def split_train_test_data(hiwy, output_dir):

    # ⭐ 날짜 컬럼 문자열 변환 (핵심 추가)
    hiwy['PRESNATN_DAYHMINSEC'] = hiwy['PRESNATN_DAYHMINSEC'].astype(int)

    # 1. 데이터 분할 (6:2:2)
    train_valid_set, test_set = train_test_split(
        hiwy, test_size=0.2, random_state=42
    )

    train_set, valid_set = train_test_split(
        train_valid_set, test_size=0.25, random_state=42
    )

    # 출력
    print("\nStep 9: Raw Data Splitting Results (6:2:2)")
    print("-" * 50)
    print(f"Total dataset:      {len(hiwy)} rows")
    print(f"Training set (60%): {len(train_set)} rows")
    print(f"Validation set (20%): {len(valid_set)} rows")
    print(f"Test set (20%):       {len(test_set)} rows")

    # ⭐ CSV 저장 (UTF-8 권장)
    train_set.to_csv(output_dir / "hiwy_train.csv", index=False, encoding="utf-8-sig")
    valid_set.to_csv(output_dir / "hiwy_valid.csv", index=False, encoding="utf-8-sig")
    test_set.to_csv(output_dir / "hiwy_test.csv", index=False, encoding="utf-8-sig")

    print("\nStep 10: Saving Files Complete")
    print("-" * 50)
    print(f"Files saved in: {output_dir}")




