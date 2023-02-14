import pandas as pd
import numpy as np
import plotly.express as px

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# apply some cool styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12, 6)

train_features = pd.read_csv("/Users/ikram/Desktop/dengAI-competition/data/dengue_features_train.csv")
train_2 = pd.read_csv("/Users/ikram/Desktop/dengAI-competition/data/dengue_features_train.csv", index_col=[0,1,2])
train_labels = pd.read_csv("/Users/ikram/Desktop/dengAI-competition/data/dengue_labels_train.csv")

test_df = pd.read_csv("/Users/ikram/Desktop/dengAI-competition/data/dengue_features_test.csv")
pt_test = test_df[['week_start_date','ndvi_ne','ndvi_nw']]
pt_test['week_start_date']=pd.to_datetime(pt_test['week_start_date'])
fig = px.line(pt_test, x=pt_test['week_start_date'], y=pt_test['ndvi_ne'])

# Add Scatter plot
fig.add_scatter(x=pt_test['week_start_date'], y=pt_test['ndvi_nw'])

# Display the plot
fig.show()

print(train_features.info())
print(train_features.isna().sum())

print(train_features['city'].unique())

train_features = train_features.drop(columns=(['year','weekofyear']),axis = 1)

grouped = train_features.groupby(train_features.city)
df_sj = grouped.get_group("sj")
df_iq = grouped.get_group("iq")

#starting with df_sj
df_sj = df_sj.drop(columns=(['city']),axis=1)

num_cols = df_sj.select_dtypes(include=np.number).columns.tolist()


corrmat = df_sj.corr()
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df_sj.columns, 
                 xticklabels=df_sj.columns, 
                 cmap="Spectral_r")
plt.show()

for col in num_cols:
    print(col)
    print('Skew :', round(sj[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    df_sj[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_sj[col])
    plt.show()

def log_transform(data,col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname]+1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    data.info()

#TODO : log transform station_precip , reanalysis_sat_precip_amt_mm

#https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/