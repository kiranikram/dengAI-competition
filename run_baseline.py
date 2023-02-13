import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from given_eda import get_train_dfs


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred, model_type):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    results = {'model_type':[model_type],
    'rmse':[rmse],
    'mae':[mae],
    'r2':[r2]}

    results_df = pd.DataFrame(results)

    with open(f"/Users/ikram/Desktop/dengAI-competition/results/baselines-{model_type}.csv", "w") as f:
        results_df.to_csv(f)

    
    return rmse, mae, r2 , results

def get_model(model_name):
    if  model_name == "GBR":
        model = GradientBoostingRegressor()
        parameters = {'learning_rate': [0.02,0.03],
                  'subsample'    : [0.9, 0.5, 0.1],
                  'n_estimators' : [100,500, 1500],
                  'max_depth'    : [6,8]
                 }

    elif model_name == "DTR":
        model = DecisionTreeRegressor()
        parameters = {"min_samples_split": [10, 20, 40],
              "max_depth": [2, 6, 8],
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100],
              }

    elif model_name == "SVR":
        


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    sj_train , iq_train = get_train_dfs()

    sj_features = sj_train.loc[:, sj_train.columns!='total_cases']
    sj_labels  = sj_train.total_cases

    iq_features = iq_train.loc[:, sj_train.columns!='total_cases']
    iq_labels  = iq_train.total_cases

    sjX_train, sjX_test, sjy_train, sjy_test = train_test_split(sj_features, sj_labels, test_size=0.30)
    iqX_train, iqX_test, iqy_train, iqy_test = train_test_split(iq_features, iq_labels, test_size=0.30)

    GBR = GradientBoostingRegressor()

    parameters = {'learning_rate': [0.02,0.03],
                  'subsample'    : [0.9, 0.5, 0.1],
                  'n_estimators' : [100,500, 1500],
                  'max_depth'    : [6,8]
                 }

    grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1)
    grid_GBR.fit(sjX_train,sjy_train)

    predicted_qualities = grid_GBR.predict(sjX_test)

    rmse, mae, r2, results = eval_metrics(sjy_test, predicted_qualities,model_type="GBR")

