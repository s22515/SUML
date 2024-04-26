import pandas as pd
import numpy as np
import math
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def prepare_cleaned_data(data_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares cleaned data for machine learning tasks by
    standardizing numerical features
    and encoding categorical features.

    Parameters:
    - data_cleaned (pd.DataFrame): DataFrame containing cleaned data.

    Returns:
    - pd.DataFrame: Prepared DataFrame ready for machine learning tasks.

    Steps:
    1. Standardization: Standardizes numerical features using StandardScaler.
    2. Encoding: Encodes categorical features using LabelEncoder.
    3. Concatenates scaled numerical features and encoded categorical features.
    """
    # standarization part

    scaler = StandardScaler()

    cols_to_scale = data_cleaned.select_dtypes(
        include=['float', 'int']).drop(columns="Age").columns

    scaled_data = scaler.fit_transform(data_cleaned[cols_to_scale])
    scaled_dataframe = pd.DataFrame(scaled_data, columns=cols_to_scale)
    scaled_dataframe = pd.concat(
        [scaled_dataframe, data_cleaned.drop(columns=cols_to_scale)],
        axis=1)
    joblib.dump(scaler, 'ml/model/scaler.bin', compress=True)

    # encoding part

    label_encoder = LabelEncoder()

    cols_to_encode = data_cleaned.select_dtypes(include=['object']).columns

    encoded_data = label_encoder.fit_transform(
        data_cleaned[cols_to_encode[0]].values
        )
    encoded_dataframe = pd.DataFrame(encoded_data, columns=cols_to_encode)
    prepared_dataframe = pd.concat(
        [encoded_dataframe, scaled_dataframe.drop(columns=cols_to_encode)],
        axis=1)

    return prepared_dataframe


def enrich_rf_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches features of the input DataFrame for Random Forest modeling.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame containing features.

    Returns:
    - pd.DataFrame:
        DataFrame with enriched features for Random Forest modeling.

    Enriched Features:
    - 'Volume': Calculated as the product of:
        'Length',
        'Height' and
        'Diameter'.
    - 'Weight proportion': Ratio of sum of:
        'Shucked Weight',
        'Viscera Weight' and
        'Shell Weight' to 'Weight'.
    - 'Shucked proportion': Ratio of 'Shucked Weight' to 'Weight'.
    - 'Viscera proportion': Ratio of 'Viscera Weight' to 'Weight'.
    - 'Shell proportion': Ratio of 'Shell Weight' to 'Weight'.
    - 'Shell area': Calculated as the area of the shell based on 'Diameter'.

    Note:
    - 'math.pi' is used for calculating the area of the shell.
    """
    dataframe["Volume"] = (
        dataframe["Length"] *
        dataframe["Height"] *
        dataframe["Diameter"]
        )
    dataframe["Weight proportion"] = (
        (
            dataframe["Shucked Weight"] +
            dataframe["Viscera Weight"] +
            dataframe["Shell Weight"]
            ) /
        dataframe["Weight"]
        )
    dataframe["Shucked proportion"] = (
        dataframe["Shucked Weight"] /
        dataframe["Weight"]
        )
    dataframe["Viscera proportion"] = (
        dataframe["Viscera Weight"] /
        dataframe["Weight"]
        )
    dataframe["Shell proportion"] = (
        dataframe["Shell Weight"] /
        dataframe["Weight"])
    dataframe["Shell area"] = (
        (dataframe["Diameter"] / 2)
        ** 2 * math.pi
        )

    return dataframe


def enrich_ridge_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches features of the input DataFrame for Ridge regression modeling.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame containing features.

    Returns:
    - pd.DataFrame:
        DataFrame with enriched features for Ridge regression modeling.

    Enriched Features:
    - 'Volume':
        Calculated as the product of 'Length', 'Height', and 'Diameter'.
    - 'Weight proportion':
        Ratio of sum of
            'Shucked Weight',
            'Viscera Weight' and
            'Shell Weight' to 'Weight'.
    - 'Shucked proportion': Ratio of 'Shucked Weight' to 'Weight'.
    - 'Viscera proportion': Ratio of 'Viscera Weight' to 'Weight'.
    - 'Shell proportion': Ratio of 'Shell Weight' to 'Weight'.

    Note:
    - No additional features specific to Ridge regression are included.
    """
    dataframe["Volume"] = (
        dataframe["Length"] *
        dataframe["Height"] *
        dataframe["Diameter"]
        )
    dataframe["Weight proportion"] = (
        (
            dataframe["Shucked Weight"] +
            dataframe["Viscera Weight"] +
            dataframe["Shell Weight"]
            ) /
        dataframe["Weight"]
        )
    dataframe["Shucked proportion"] = (
        dataframe["Shucked Weight"] /
        dataframe["Weight"]
        )
    dataframe["Viscera proportion"] = (
        dataframe["Viscera Weight"] /
        dataframe["Weight"]
        )
    dataframe["Shell proportion"] = (
        dataframe["Shell Weight"] /
        dataframe["Weight"])

    return dataframe


def split_data(
    prepared_dataframe: pd.DataFrame,
    target_name: str,
    in_test_size=0.35,
    in_random_state=0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the prepared DataFrame into
    features and target variables for training and testing.

    Parameters:
    - prepared_dataframe (pd.DataFrame):
        DataFrame containing prepared data.
    - target_name (str):
        Name of the target variable column.
    - in_test_size (float):
        The proportion of the dataset to include in the test split.
    - in_random_state (int):
        Controls the randomness of the training and testing indices.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Tuple containing X_train, X_test, y_train, y_test.

    Note:
    - X_train: Features for training.
    - X_test: Features for testing.
    - y_train: Target variable for training.
    - y_test: Target variable for testing.
    """
    feature = prepared_dataframe.drop(columns=[target_name])
    target = prepared_dataframe[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        feature,
        target,
        test_size=in_test_size,
        random_state=in_random_state)

    return X_train, X_test, y_train, y_test
