import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

df = pd.read_parquet("cmm_erdos_bootcamp_2020_timeseries.pq", engine='pyarrow')
df.date_val = pd.to_datetime(df.date_val)
df['series'] = df['volume_A']
order=(1,0,2)
seasonal_order=(1,0,2,7)
exog_feat = ['is_holiday']

print(order, seasonal_order, exog_feat)

#############################################################################

train_test_sep = datetime(2019, 1, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

train_ = df_train.series.values
test_ = df_test.series.values
np.disp(len(test_))

exog_train = df_train[exog_feat]
exog_test = df_test[exog_feat]

model = SARIMAX(train_, order=order, seasonal_order=seasonal_order, exog=exog_train)
fit_ = model.fit(disp=False, method='powell')

forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)

 
np.disp(np.abs((test_ - forecast_) / test_).mean())

###########################################################################

train_test_sep = datetime(2019, 7, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

train_ = df_train.series.values
test_ = df_test.series.values
np.disp(len(test_))

exog_train = df_train[exog_feat]
exog_test = df_test[exog_feat]

model = SARIMAX(train_, order=order, seasonal_order=seasonal_order, exog=exog_train)
fit_ = model.fit(disp=False, method='powell')

forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)

np.disp(np.abs((test_ - forecast_) / test_).mean())

#############################################################################3

train_test_sep = datetime(2019, 10, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

train_ = df_train.series.values
test_ = df_test.series.values
np.disp(len(test_))

exog_train = df_train[exog_feat]
exog_test = df_test[exog_feat]

model = SARIMAX(train_, order=order, seasonal_order=seasonal_order, exog=exog_train)
fit_ = model.fit(disp=False, method='powell')

forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)
 
np.disp(np.abs((test_ - forecast_) / test_).mean())

#########################################################################3##

train_test_sep = datetime(2019, 12, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

train_ = df_train.series.values
test_ = df_test.series.values
np.disp(len(test_))

exog_train = df_train[exog_feat]
exog_test = df_test[exog_feat]

model = SARIMAX(train_, order=order, seasonal_order=seasonal_order, exog=exog_train)
fit_ = model.fit(disp=False, method='powell')

forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)
 
np.disp(np.abs((test_ - forecast_) / test_).mean())


 