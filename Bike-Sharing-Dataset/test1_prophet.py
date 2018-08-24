# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:11:41 2018

@author: durairaj
"""

import pandas as pd
df=pd.read_csv('day.csv',index_col=0)
cnt_ninty=df['cnt'].quantile(0.90)
cnt_ten=df['cnt'].quantile(0.10)
df.drop(df[(df.cnt<cnt_ten)&(df.cnt<cnt_ninty)].index,inplace=True)
dfp=df[['dteday','cnt']]
dfp['dteday']=pd.to_datetime(dfp['dteday'])
#print(dfp.head(10))
from fbprophet import Prophet
p=Prophet()
dfp=dfp.rename(columns={'dteday':'ds','cnt':'y'})
p.fit(dfp)
future_cnt=p.make_future_dataframe(periods=365)
forecast_cnt=p.predict(future_cnt)
#print(forecast_cnt.head())
p.plot(forecast_cnt)
p.plot_components(forecast_cnt) 