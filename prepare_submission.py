import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from functools import partial, reduce

def split_data_by_city(df):
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj , iq

test_data = pd.read_csv("/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/dengue_features_test.csv",index_col=[0, 1, 2])
test_data = test_data[['reanalysis_relative_humidity_percent','station_avg_temp_c','station_min_temp_c','reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k']]

#retrain model on all training data 

train_features = pd.read_csv("/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/dengue_features_train.csv",index_col=[0, 1, 2])
train_features = train_features[['reanalysis_relative_humidity_percent','station_avg_temp_c','station_min_temp_c','reanalysis_specific_humidity_g_per_kg','reanalysis_dew_point_temp_k']]

train_labels = pd.read_csv("/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/dengue_labels_train.csv",index_col=[0, 1, 2])

sj_train , iq_train = split_data_by_city(train_features)
sj_label , iq_label = split_data_by_city(train_labels)
sj_test , iq_test = split_data_by_city(test_data)

model = SVR()
params = {
            "kernel": ("linear", "rbf", "poly"),
            "C": [1.5, 10],
            "gamma": [1e-7, 1e-4],
            "epsilon": [0.1, 0.2, 0.5, 0.3],
        }

grid_model_sj = GridSearchCV(estimator=model, param_grid=params, cv=2)
grid_model_sj.fit(sj_train, sj_label)
sj_predictions = grid_model_sj.predict(sj_test)

grid_model_iq = GridSearchCV(estimator=model, param_grid=params, cv=2)
grid_model_iq.fit(iq_train, iq_label)
iq_predictions = grid_model_iq.predict(iq_test)

#add predictions to test set for feature engineering 

sj_test['total_cases'] = sj_predictions
iq_test['total_cases'] = iq_predictions

#prepare traning dataset

train_features = train_features.join(train_labels)

avg_4 = add_4_rolling_means(train_features)
avg_8 = add_8_rolling_means(train_features)
_max = add_previous_max(train_features)

dfs = [train_features, avg_4, avg_8,_max]
merge = partial(pd.merge, on=['year','weekofyear'], how='outer')
lagged_df = reduce(merge, dfs)
df_labels = lagged_df.total_cases
lagged_df = lagged_df.loc[:, lagged_df.columns != "total_cases"]

#TODO: drop columns that are not needed ? 

lagged_df = lagged_df.fillna(method='bfill')

sj_lagged , iq_lagged = split_data_by_city(lagged_df)
sj_labels , iq_labels = split_data_by_city(df_labels)

#prepare test dataset 

#SJ dataset

avg_4 = add_4_rolling_means(sj_test)
avg_8 = add_8_rolling_means(sj_test)
_max = add_previous_max(sj_test)

dfs = [sj_test, avg_4, avg_8,_max]
merge = partial(pd.merge, on=['year','weekofyear'], how='outer')
test_sj = reduce(merge, dfs)
test_sj = test_sj.loc[:, test_sj.columns != "total_cases"]

#IQ dataset

avg_4 = add_4_rolling_means(iq_test)
avg_8 = add_8_rolling_means(iq_test)
_max = add_previous_max(iq_test)

dfs = [iq_test, avg_4, avg_8,_max]
merge = partial(pd.merge, on=['year','weekofyear'], how='outer')
test_iq = reduce(merge, dfs)
test_iq = test_iq.loc[:, test_iq.columns != "total_cases"]

retrain_model_sj = SVR()
retrain_model_iq = SVR()

retrain_model_sj.fit(sj_lagged, sj_labels )
retrain_model_sj.fit(sj_lagged, sj_labels )


sj_predictions_final = retrain_model_sj.predict(test_sj)
iq_predictions_final = retrain_model_iq.predict(test_iq)

#concat sj and iq predictions
all_predictions = sj+iq

submission = pd.read_csv("/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/submission_format.csv")

submission["total_cases"] = all_predictions