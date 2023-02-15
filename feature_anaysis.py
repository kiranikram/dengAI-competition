import os
import sys
os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functools import partial, reduce


from given_eda import get_train_dfs , split_data_by_city , add_4_rolling_means
from feature_eng import add_4_rolling_means,add_8_rolling_means,add_previous_max,add_dengue_lags

all_data = get_train_dfs()

sj , iq = split_data_by_city(all_data)

sj_labels = sj.total_cases
sj_4 = add_4_rolling_means(sj)
sj_8 = add_8_rolling_means(sj)
sj_max = add_previous_max(sj)

dfs = [sj, sj_4, sj_8,sj_max]
merge = partial(pd.merge, on=['year','weekofyear'], how='outer')
lagged_sj = reduce(merge, dfs)
lagged_sj = lagged_sj.loc[:, lagged_sj.columns != "total_cases"]

lagged_sj = lagged_sj.fillna(method='bfill')

sjX_train, sjX_test, sjy_train, sjy_test = train_test_split(
        lagged_sj, sj_labels, test_size=0.30
    )

regressor = RandomForestRegressor()
regressor.fit(sjX_train, sjy_train)

features = lagged_sj.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()



