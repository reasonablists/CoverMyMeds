#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:46:41 2020

@author: vasudha
"""


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
df['series'] = df['volume_A'] #enter the series you want to test

train_test_sep = datetime(2019, 1, 1)
df_train = df.loc[df.date_val < train_test_sep]
df_test = df.loc[df.date_val >= train_test_sep]

exog_feat = ['is_holiday'] 

orders = [    (1,0,0),
    (2,0,0),
    (1,1,0),

    ]
s_orders = [
    (1,0,0,7),
    (2,0,0,7),

    ]



results = np.ones((len(orders),len(s_orders)))


for i in range(0,len(orders)):
    for j in range(0,len(s_orders)):
        results_temp = np.ones((12))
        for k in range(1, 12):
            try:
                
                cv_sep = datetime(2018, k, 1)
                df_train_train = df_train.loc[df_train.date_val < cv_sep]
                temp = df_train.loc[df_train.date_val >= cv_sep]
                df_train_test = temp.loc[temp.calendar_month==k]
    
                train_ = df_train_train.series.values
                test_ = df_train_test.series.values
                #np.disp(len(test_))
    
                exog_train = df_train_train[exog_feat]
                exog_test = df_train_test[exog_feat]
    
                model = SARIMAX(train_, order=orders[i], seasonal_order=s_orders[j], exog=exog_train)
                fit_ = model.fit(disp=False, method='powell')
    
                forecast_ = fit_.forecast(len(test_), exog=exog_test).reset_index(drop=True)
    
                results_temp[k-1] = (np.abs((test_ - forecast_) / test_).mean())
    
    
            except Exception:
                pass
            
        results[i,j] = np.min(results_temp)
        
        

index_array = np.argmin(results, axis=-1)
print(orders[index_array[0]], s_orders[index_array[1]])
        
        
        
        
        
        