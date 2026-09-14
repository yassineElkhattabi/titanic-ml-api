import pandas as pd


def basic_eda(df: pd.DataFrame) -> None:
    print("\n--- DATASET SHAPE ---")
    print(df.shape)

    print("\n--- COLUMNS ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATES ---")
    print(df.duplicated().sum())

    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe(include="all"))