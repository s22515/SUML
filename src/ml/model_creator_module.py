import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def create_ridge_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    in_alpha=1.0,
    in_fit_intercept=True,
    in_solver='auto'
        ) -> Ridge:
    """
    Creates a Ridge regression model.

    Parameters:
    - X_train (np.ndarray): Features for training.
    - y_train (np.ndarray): Target variable for training.
    - in_alpha (float): Regularization strength (default is 1.0).
    - in_fit_intercept (bool):
        Whether to calculate the intercept for this model (default is True).
    - in_solver (str): Solver to use for fitting the model (default is 'auto').

    Returns:
    - Ridge: Ridge regression model.

    Note:
    - The model is fitted using the provided training data.
    """
    linear_model = Ridge(
        alpha=in_alpha,
        fit_intercept=in_fit_intercept,
        solver=in_solver
        )
    linear_model.fit(X_train, y_train)
    return linear_model


def create_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    in_max_depth=None,
    in_max_features=1.0,
    in_n_estimators=100,
    in_min_samples_split=2,
    in_min_samples_leaf=1
        ) -> RandomForestRegressor:
    """
    Creates a Random Forest regression model.

    Parameters:
    - X_train (np.ndarray): Features for training.
    - y_train (np.ndarray): Target variable for training.
    - in_max_depth (Optional[int]):
        Maximum depth of the tree (default is None).
    - in_max_features (Union[int, float]):
        The number of features to consider when looking for the best split.
    - in_n_estimators (int): Number of trees in the forest.
    - in_min_samples_split (int):
        Minimum number of samples required to split an internal node.
    - in_min_samples_leaf (int):
        Minimum number of samples required to be at a leaf node.

    Returns:
    - RandomForestRegressor: Random Forest regression model.

    Note:
    - The model is fitted using the provided training data.
    """
    rf_model = RandomForestRegressor(
        max_depth=in_max_depth,
        max_features=in_max_features,
        n_estimators=in_n_estimators,
        min_samples_leaf=in_min_samples_leaf,
        min_samples_split=in_min_samples_split
        )
    rf_model.fit(X_train, y_train)
    return rf_model


def save_model(model: RandomForestRegressor) -> None:
    """
    Saves the trained Random Forest regression model to a file.

    Parameters:
    - model (RandomForestRegressor): Trained Random Forest regression model.

    Returns:
    - None

    Note:
    - The model is saved to the 'model/rf_model.sav' file.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'model/rf_model.sav')
    joblib.dump(model, filename)


def load_model() -> RandomForestRegressor:
    """
    Loads the trained Random Forest regression model from a file.

    Returns:
    - RandomForestRegressor: Loaded Random Forest regression model.

    Note:
    - The model is loaded from the 'model/rf_model.sav' file.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'model/rf_model.sav')
    model = joblib.load(filename)
    return model


def get_model_prediction(input_data: np.ndarray) -> float:
    """
    Gets predictions from the loaded Random Forest regression model.

    Parameters:
    - input_data (np.ndarray): Input data for making predictions.

    Returns:
    - float: Prediction made by the model.

    Note:
    - The Random Forest regression model
        is loaded using the 'load_model' function.
    """
    model = load_model()
    prediction = model.predict(input_data)
    return prediction
