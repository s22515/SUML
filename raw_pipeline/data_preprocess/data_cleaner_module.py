import pandas as pd


def clean_Data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing rows based on specified conditions.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame containing the data to be cleaned.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with rows removed based on specified conditions.
    """
    if len(dataframe) == 0:
        print("Error: The data is empty.")
        return None

    dataframe = dataframe[dataframe['Viscera Weight'] < 17.5]
    dataframe = dataframe[dataframe['Shell Weight'] <= 24]
    dataframe = dataframe[dataframe['Shucked Weight'] <= 36]
    dataframe = dataframe[dataframe['Weight'] <= 78]
    dataframe = dataframe[dataframe['Height'] < 1]

    return dataframe
