import pandas as pd

path_to_features = "/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/dengue_features_train.csv"
path_to_labels = "/Users/mussayababbasshamsi/Desktop/Kiran/dengAI-competition/data2/dengue_labels_train.csv"

def get_train_dfs():
    train_df = preprocess_data(path_to_features, labels_path = path_to_labels)
    return train_df


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    #now including more than the original 4. remember: baselines were created using original 4
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'reanalysis_relative_humidity_percent',
                 'reanalysis_min_air_temp_k',
                 'reanalysis_max_air_temp_k'
                 ]
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    return df

def split_data_by_city(df):
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj , iq

def add_4_rolling_means(df):

    df2 = df.rolling(window=4).mean()
    df2 = df2.rename(columns={x: x + '_4' for x in df2.columns})
    df_all = pd.merge(df,df2, how='left',on=['year','weekofyear'])

    return df_all





    
   



