from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import math

# Input: model, 2 numpy arrays that contains test feature and targets
# Output: Model predictions, metrics: mae, mse, rmse, r2
def get_model_metrics(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:

    model_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, model_predict)
    mse = mean_squared_error(y_test, model_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, model_predict)

    return model_predict, mae, mse, rmse, r2

def get_model_metrics(model: Ridge, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:

    model_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, model_predict)
    mse = mean_squared_error(y_test, model_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, model_predict)

    return model_predict, mae, mse, rmse, r2

# Input: model, 2 numpy arrays that contains train feature and targets
# Output: Cross validation scores, mean and devation
def get_cross_validation_metrics(model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:

    scores = cross_val_score(model, X_train, y_train, cv=5)

    return scores, scores.mean(), scores.std()

def get_cross_validation_metrics(model: Ridge, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:

    scores = cross_val_score(model, X_train, y_train, cv=5)

    return scores, scores.mean(), scores.std()

# Input: model parameter as dictionary, model, train feature, train target, iteration, cross validation iter, random state
# Output: string -> best params
def model_params_random_search(params: dict, model, X_train: np.ndarray, y_train: np.ndarray, in_n_iter=8, in_cv=6, in_random_state=0):

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=in_n_iter, cv=in_cv, random_state=in_random_state, error_score='raise')
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_