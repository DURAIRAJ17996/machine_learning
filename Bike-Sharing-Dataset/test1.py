# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 08:41:23 2018

@author: durairaj
"""
#['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
# 'workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed',
 #'casual','registered', 'cnt']
import pandas as pd
df=pd.read_csv('day.csv',index_col=0)
#print(df['dteday'].head())
df_date=df['dteday']
df.drop(['casual','registered','dteday'],inplace=True,axis=1)
#print(df.columns)
print(df.shape)
cnt_ninty=df['cnt'].quantile(0.95)
cnt_ten=df['cnt'].quantile(0.005)
df.drop(df[(df.cnt<cnt_ten)|(df.cnt>cnt_ninty)].index,inplace=True)
print(df.shape)
from sklearn.preprocessing import OneHotEncoder
hotencode=OneHotEncoder(categorical_features=[0])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[4])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[6])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[18])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[20])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[27])
df=hotencode.fit_transform(df).toarray()

hotencode=OneHotEncoder(categorical_features=[29])
df=hotencode.fit_transform(df).toarray()
df_final=pd.DataFrame(df)
#print(df_final.head())

X=df_final.iloc[:,:-1]
y=df_final.iloc[:,-1]
#print(X.shape,y.shape)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X)
from sklearn.metrics import r2_score
print("Accuracy::",r2_score(y,y_pred))
print(y_pred,y)
#X_model=[[0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0.44,0.4394,0.88,0.3582]]
#X_model=[[0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0.233333,0.248112,0.49875	,0.157963]]
#print(len(X_model))
#X_model=pd.DataFrame(X_model)
#y_pred_model=lr.predict(X_model)
#print("cnt ",y_pred_model)
#import matplotlib.pyplot as plt
#import numpy as np
#plt.scatter((np.arange(len(y))),y)
from sklearn.feature_selection import RFE
rfe=RFE(lr,5)
fit=rfe.fit(X,y)
print(format(fit.ranking_))