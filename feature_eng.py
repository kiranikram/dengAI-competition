import pandas as pd 

#mmake some lags 
def add_4_rolling_means(df):

    df2 = df.rolling(window=4).mean()
    df2 = df2.rename(columns={x: x + '_4' for x in df2.columns})
    df_all = pd.merge(df,df2, how='left',on=['year','weekofyear'])

    return df_all

def add_8_rolling_means(df):

    df2 = df.rolling(window=8).mean()
    df2 = df2.rename(columns={x: x + '_8' for x in df2.columns})
    df_all = pd.merge(df,df2, how='left',on=['year','weekofyear'])

    return df_all


def  add_previous_max(df):
    df2 = df.rolling(window=4).max()
    df2 = df2.rename(columns={x: x + '_max' for x in df2.columns})
    df_all = pd.merge(df,df2, how='left',on=['year','weekofyear'])

    return df_all

def add_dengue_lags(df):

    df['prev_avg_cases_4'] = df['total_cases'].rolling(window=4).mean()
    df['prev_avg_cases_8'] = df['total_cases'].rolling(window=8).mean()

    return df