import pandas as pd
import os


def import_dataframe_from_csv(path: str) -> pd.DataFrame:
    """
    Imports a DataFrame from a CSV file located at the specified path.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame:
        DataFrame imported from the CSV file if successful,
        otherwise an empty DataFrame.
    """
    try:
        print(os.getcwd())
        imported_data = pd.read_csv(path)
        return imported_data

    except FileNotFoundError:
        print(f"Error: File not found in '{path}'")
        return pd.DataFrame()

    except Exception as exp:
        print(f"Error: {exp}")
        return None
