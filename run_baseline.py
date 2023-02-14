import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from given_eda import get_train_dfs


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred, model_type, city):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    results = {
        "city": [city],
        "model_type": [model_type],
        "rmse": [rmse],
        "mae": [mae],
        "r2": [r2],
    }

    print(f"running city:{city} with model {model_type}")

    results_df = pd.DataFrame(results)

    with open(
        f"/Users/ikram/Desktop/dengAI-competition/results/baselines-{city}-{model_type}.csv",
        "w",
    ) as f:
        results_df.to_csv(f)

    return rmse, mae, r2, results


def get_model(model_name):
    if model_name == "GBR":
        model = GradientBoostingRegressor()
        parameters = {
            "learning_rate": [0.02, 0.03],
            "subsample": [0.9, 0.5, 0.1],
            "n_estimators": [100, 500, 1500],
            "max_depth": [6, 8],
        }

    elif model_name == "DTR":
        model = DecisionTreeRegressor()
        parameters = {
            "min_samples_split": [10, 20, 40],
            "max_depth": [2, 6, 8],
            "min_samples_leaf": [20, 40, 100],
            "max_leaf_nodes": [5, 20, 100],
        }

    elif model_name == "SVR":
        model = SVR()
        parameters = {
            "kernel": ("linear", "rbf", "poly"),
            "C": [1.5, 10],
            "gamma": [1e-7, 1e-4],
            "epsilon": [0.1, 0.2, 0.5, 0.3],
        }

    return model, parameters

def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(xdata, ydata)(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    models = ["SVR", "DTR", "GBR"]

    sj_train, iq_train = get_train_dfs()

    sj_features = sj_train.loc[:, sj_train.columns != "total_cases"]
    sj_labels = sj_train.total_cases

    iq_features = iq_train.loc[:, sj_train.columns != "total_cases"]
    iq_labels = iq_train.total_cases

    sjX_train, sjX_test, sjy_train, sjy_test = train_test_split(
        sj_features, sj_labels, test_size=0.30
    )
    iqX_train, iqX_test, iqy_train, iqy_test = train_test_split(
        iq_features, iq_labels, test_size=0.30
    )

    for regression_model in models:
        model, params = get_model(regression_model)
        grid_model_sj = GridSearchCV(estimator=model, param_grid=params, cv=2)
        grid_model_iq = GridSearchCV(estimator=model, param_grid=params, cv=2)
        grid_model_sj.fit(sjX_train, sjy_train)
        grid_model_iq.fit(iqX_train, iqy_train)

        sj_predictions = grid_model_sj.predict(sjX_test)
        iq_predictions = grid_model_iq.predict(iqX_test)

        rmse, mae, r2, results = eval_metrics(
            sjy_test, sj_predictions, model_type=regression_model, city="SJ"
        )
        rmse, mae, r2, results = eval_metrics(
            iqy_test, iq_predictions, model_type=regression_model, city="IQ"
        )
