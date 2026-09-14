import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset and return it as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)

        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"Error: file not found -> {file_path}")
        raise

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise