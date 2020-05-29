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
df['series'] = df['volume_B']

df = df.loc[df.date_val >= datetime(2018, 1, 1)]

train_test_sep = datetime(2019, 1, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

train_ = df_train.series.values
test_ = df_test.series.values

exog_train = df_train[['is_holiday']]
exog_test = df_test[['is_holiday']]

model = SARIMAX(train_, order=(1, 0, 2), seasonal_order=(2, 1, 1, 7), exog=exog_train)
fit_ = model.fit(disp=False, method='powell')

forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)

fig,ax = plt.subplots(figsize = (16, 5))
ax.plot(test_ , label='test')
ax.plot(forecast_ , label='forecast')
plt.legend()
plt.show()
 
np.disp(np.abs((test_ - forecast_) / test_).mean())

fig,ax = plt.subplots(figsize = (16, 5))
ax.plot(test_[0:30] , label='test')
ax.plot(forecast_[0:30] , label='forecast')
plt.legend()
plt.show()
 
np.disp(np.abs((test_[0:30] - forecast_[0:30]) / test_[0:30]).mean())
np.disp(np.abs((test_[0:182] - forecast_[0:182]) / test_[0:182]).mean())