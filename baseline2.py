#!/usr/bin/python
# this script creates a baseline submission using Gaussian Process Regression
# it will be properly commented later


import pandas as pd
import numpy as np
import datetime as dt
import os

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 10000) 
pd.set_option('display.max_columns', 60)

dpath = "../bikesharing"

def datetime_parser(ts):
  if type(ts) == str:
    return dt.datetime.strftime(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
  else:
    return np.nan

def noise(x):
  return np.float(np.random.normal(0,1,1))

def read_bike_csv(data):
  df = pd.read_csv(data, 
    dtype={
     'datetime'   : str,
     'season'     : np.integer,
     'holiday'    : np.integer,
     'workingday' : np.integer,
     'weather'    : np.integer,
     'temp'       : np.floating,
     'atemp'      : np.floating,
     'humidity'   : np.floating,
     'windspeed'  : np.floating,
     'casual'     : np.integer,
     'registered' : np.integer,
     'count'      : np.integer})
  df['datetime'] = map(datetime_parser, df['datetime'])
  df['hour'] = df['datetime'].apply(lambda x: pd.to_datetime(x).hour) 
  df['month'] = df['datetime'].apply(lambda x: pd.to_datetime(x).month)
  df['year'] = df['datetime'].apply(lambda x: pd.to_datetime(x).year) 
  return(df)

train = read_bike_csv(os.path.join(dpath,'data/train.csv'))
test  = read_bike_csv(os.path.join(dpath,'data/test.csv'))
count = []

train['temp'] = map(noise, train['temp'])
train['atemp'] = map(noise, train['atemp'])
train['humidity'] = map(noise, train['humidity'])
train['windspeed'] = map(noise, train['windspeed'])

#### RANDOM FOREST ########

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


features = ['holiday','workingday', 'temp','atemp','humidity','hour']
param_grid = { 
    'n_estimators': [100, 250, 800, 1000, 1200, 1500],
    'max_features': [None, 'sqrt', 'log2']
}


param_grid = { 
    'n_estimators': [100, 500, 1000, 1500, 2000],
    'max_features': [None, 'sqrt', 'log2']
}

for year in test.year.unique():
    for month in test.month.unique():
        testset = test.loc[(test['year'] == year) & (test['month'] == month)] 
        features = ['holiday','workingday', 'temp', 'atemp', 'humidity', 'hour']
        trainset = train.loc[(train['datetime'] <= testset['datetime'].min())]
        rf = RandomForestRegressor(n_jobs=-1, max_features= None, 
                                   n_estimators=50, oob_score = True, 
                                   max_depth=None, min_samples_split=1) 
        rfcv = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
        casual = rfcv.fit(trainset[features],trainset['casual'])
        ypred_cas = casual.predict(testset[features])
        #### 
        rf = RandomForestRegressor(n_jobs=-1, max_features= None, 
                                   n_estimators=50, oob_score = True, 
                                   max_depth=None, min_samples_split=1) 
        rfcv = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5) 
        registered = rfcv.fit(trainset[features],trainset['registered'])
        ypred_reg = registered.predict(testset[features])
        count = np.append(count,(ypred_cas + ypred_reg).astype(int))
        print "year: %s, month: %s, casual: %s, registered: %s" % (str(year), str(month), 
                                            str(casual.best_params_), str(registered.best_params_))


subm = pd.concat([test.datetime, pd.Series(count)],axis=1)      
subm.columns = ['datetime','count']
subm.to_csv(os.path.join(dpath,'to_submit_rf.csv'),index=False)



#### GPR ######

'''

#"season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "hour"
from sklearn.gaussian_process import GaussianProcess
for year in test.year.unique():
    for month in test.month.unique():
        testset = test.loc[(test['year'] == year) & (test['month'] == month)]
        features = ['temp', 'atemp', 'humidity','windspeed','hour']
        trainset = train.loc[(train['datetime'] <= testset['datetime'].min())]
        xtrain = train[features].values
        ytrain = train['count'].ravel()
        xtest = test[features].values
        gp = GaussianProcess()
        gp.fit(xtrain,ytrain)
        ypred = gp.predict(xtest)    
        count = np.append(count,ypred)

subm = pd.concat([test.datetime, pd.Series(count)],axis=1)        
subm.columns = ['datetime','count']
subm.to_csv(os.path.join(dpath,'to_submit_gp.csv'))

'''
