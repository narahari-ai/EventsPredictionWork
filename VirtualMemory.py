# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:05:21 2020

@author: GJU5KOR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
from utility import EmailFormatUtility
import random


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,train_test_split,KFold,LeaveOneOut
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso



from sklearn.svm import OneClassSVM,SVR

#
#df = pd.concat( pd.read_excel("data/Final.xlsx",sheet_name=None),ignore_index=True)
#
#back_df = df.copy()
#df=back_df.copy()

df = pd.read_csv("data/virtualMemoryData.csv",parse_dates=["Time"],index_col=0 )

df = df[df["Alert Classification"]=="virtual memory usage"]
df.reset_index(drop=True,inplace=True)

seed_value= 22 

arycolsDrop = [col for col in df.columns if (sum(df[col].isnull())== df.shape[0])]
df.drop(columns=arycolsDrop,inplace=True)
constant_features = [
    feat for feat in df.columns if len(df[feat].unique()) == 1
]
df.drop(columns=constant_features,inplace=True )

def modifyDateTimeField(dt):
    if(dt):
        if(dt.find('.')>-1):
            arydata = dt.split('.')
            dt = arydata[1]+"/"+arydata[0]+"/"+arydata[2]
    return pd.to_datetime(dt)



df["Time"] = df["Time of Event"] .map(modifyDateTimeField)
df["ServerName"] = df.ServerName.map(lambda x: x if x.find('.')== -1 else x.split('.')[0])
df["KSName"] = df['KS Name'].map(lambda x: x.split(':')[0].strip() if x.find(':')>-1 else x.strip() )
df["MemoryUsed"] = df["Process Name"].map(EmailFormatUtility.getonlyMemoryUsed)

df.Severity.value_counts()


df = df[["Time","ServerName","MemoryUsed"]]


df = df.rename(columns={"ServerName":'Server'})
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = EmailFormatUtility.extractInfoFromDates(df)
df.sort_values(by='Time',inplace=True)
df.reset_index(drop=True,inplace=True) 


def GetDataForTopCountRecords(df,ServerName):
    ds = df[df["Server"] == ServerName]
    ds = ds.drop_duplicates("Time",keep='last')
    ds = ds.set_index("Time")
    ds.sort_index()
    return ds


def getAllHoursofDay(curDate=datetime.today()+ timedelta(days=1)):
    curDate = curDate.replace(hour=0,microsecond=0,second=0,minute=0)
    s = pd.Series(pd.date_range(start=curDate,periods=24,freq='h'))
    ts = pd.DataFrame({"Time": s})
    ts = EmailFormatUtility.extractInfoFromDates(ts)
    ts = ts.set_index("Time")
#    ts.drop(columns=["Time"],inplace=True)
    return ts
def TrainTestSplitTimeData(df,split_date=pd.datetime(2020,2,3)):
    df = df.drop(columns=["Server"])
    train_df = df.loc[:split_date]
    test_df = df[split_date:]
    X_train = train_df.drop(columns=['MemoryUsed'])
    y_train = train_df["MemoryUsed"]
    X_test = test_df.drop(columns=['MemoryUsed'])
    y_test = test_df["MemoryUsed"]
    return X_train,X_test,y_train,y_test

def TrainTestSplitTimeDataNew(df,start_date,split_date=pd.datetime(2020,2,3)):
    df = df.drop(columns=["Server"])
    if(not pd.isna(start_date)):
        train_df = df.loc[start_date:split_date]
    test_df = df[split_date:]
    X_train = train_df.drop(columns=['MemoryUsed'])
    y_train = train_df["MemoryUsed"]
    X_test = test_df.drop(columns=['MemoryUsed'])
    y_test = test_df["MemoryUsed"]
    return X_train,X_test,y_train,y_test


def workonETRmodel(X_train,y_train,X_test,y_test,df_result):
    model = ExtraTreesRegressor(random_state=seed_value)
    res = model.fit(X_train,y_train)
    y_pred = res.predict(X_test)
    print("Mean absolute error: ",EmailFormatUtility.evaluate(model,X_test,y_test))
    rms = np.sqrt(mse(y_test, y_pred)) 
    print("Root means Square Error:",rms)
    df_result["result"] = y_pred
    y_pred = y_pred.reshape(-1,1)
    return res


def showPredictionValidation(y_train,y_test,X_test,X_valid,df_result):
    plt.figure(figsize=(16,8))
    plt.title(serverName)
    plt.plot(pd.DataFrame(y_train), label='Train') 
    plt.plot(pd.DataFrame(y_test), label='Valid') 
    plt.plot(pd.DataFrame(df_result["result"]), label='Prediction') 
    plt.plot(pd.DataFrame(X_valid["Prediction"]), label='Future') 
    plt.legend(loc='best') 
    plt.show()
    
def formatTimeval(t):
    retdt = t+timedelta(minutes=-t.minute,seconds=-t.second)
    return retdt

df["Time"] = df["Time"].map(formatTimeval)

df.drop_duplicates(inplace=True)


def getDatesBetween(start=None,end=None):
    start = start+timedelta(hours=-start.hour,minutes=-start.minute,seconds=-start.second)
    diffDates = end-start
    ret = pd.Series()
    for i in range(diffDates.days+1):
        day = start+timedelta(days=i)
        s = pd.Series(pd.date_range(start=day,periods=24,freq='h'))
        ret = ret.append(s)
    ts = pd.DataFrame({"Time": ret})
#    ts = ts.set_index("Time")
#    ts["MemoryUsed"] = ts.index.map(getUsageByTime)
    return ts
def getFutureDates(maxDate):
    s = pd.Series(pd.date_range(start=maxDate,periods=15,freq="d"))
    ts = pd.DataFrame({"Time": s})
    ts = EmailFormatUtility.extractInfoFromDates(ts)
    return ts.set_index("Time")

def getUsageByTime(t):
#    v =  round(random.uniform(70,70),2)
    v = 50.00
    if(t in ds.index):
        v = ds.loc[t]["MemoryUsed"]
    print(v)   
    return v

def get_part_of_day(hour):
    return (
        0 if 5 <= hour <= 11
        else
        1 if 12 <= hour <= 17
        else
        2 if 18 <= hour <= 22
        else
        3
    )
    
def getWorkingorNonWorkingHoursOfDay(hour):
    return (0 if  8 <=  hour <=18 else 1)






    
serverName = df.Server.value_counts().keys()[0]
ds = GetDataForTopCountRecords(df,serverName)

#ds = ds.index.drop_duplicates()
#ds = pd.DataFrame(ds)
#ds = ds.set_index("Time")


ds1 = getDatesBetween(ds.index.min(),ds.index.max())
ds1.sort_values(by='Time',inplace=True)
ds1.reset_index(drop=True,inplace=True)

#ds1 = ds1.set_index("Time")
#ds1 = pd.DataFrame(ds1)

ds1 = ds1.drop_duplicates('Time',keep='last')


ds1["MemoryUsed"] = ds1.Time.map(getUsageByTime)


#ds1["MemoryUsed"] = ds1.Time.map(lambda x: ds.loc[x]["MemoryUsed"] if x in ds.index else  round(random.uniform(50,51),2))


ds1["Server"] = serverName
ds1 = EmailFormatUtility.extractInfoFromDates(ds1)

ds1["PartOfDay"] = ds1.Hour.map(get_part_of_day)
ds1["IsWorkingTime"] = ds1.Hour.map(getWorkingorNonWorkingHoursOfDay)

ds1 = ds1.set_index("Time")

maxDate = ds.index.max()
print(ds.index.min())
print(ds.index.max())

#ts = getAllHoursofDay()

print(ds1.index.min())
print(ds1.index.max())


ts = getAllHoursofDay()

#dt = pd.datetime(2020,2,1)


#X_train,X_test,y_train,y_test = TrainTestSplitTimeData(ds1,pd.datetime(2020,2,1))


X_train,X_test,y_train,y_test = TrainTestSplitTimeDataNew(ds1,pd.datetime(2019,6,1),pd.datetime(2020,1,1))



df_result = pd.DataFrame(index=X_test.index.copy())

model =  workonETRmodel(X_train,y_train,X_test,y_test,df_result)
y_pred = pd.DataFrame(index=X_test.index.copy())
y_pred["result"] = model.predict(X_test)

X_valid = getFutureDates(maxDate)
X_valid["PartOfDay"] = X_valid.Hour.map(get_part_of_day)
X_valid["IsWorkingTime"] = X_valid.Hour.map(getWorkingorNonWorkingHoursOfDay)

X_valid["Prediction"] = model.predict(X_valid)

showPredictionValidation(y_train,y_test,X_test,X_valid,df_result)

print(mae(y_test,y_pred))


print(mse(y_test,y_pred))


print(r2(y_test,y_pred))




pipelines = []
# =============================================================================

pipelines.append(('DSTR', DecisionTreeRegressor()))
pipelines.append(('GBM', GradientBoostingRegressor()))
pipelines.append(('RDMF', RandomForestRegressor()))
pipelines.append(('ADAB', AdaBoostRegressor()))
pipelines.append(('ETR', ExtraTreesRegressor()))
pipelines.append(('BAGR', BaggingRegressor()))
pipelines.append(('KNNR', KNeighborsRegressor(n_neighbors = 7)))
#pipelines.append(('LR', LinearRegression()))
#pipelines.append(('Ridge', Ridge()))
#pipelines.append(('Lasso', Lasso()))
#pipelines.append(('SVR', SVR()))

## =============================================================================


def apply_loocv(X_train,y_train,X_test,y_test):
    dict = {}
    results = []
    names = []
    for name, model in pipelines:
        loocv = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=loocv, scoring='neg_mean_absolute_error')
        model.fit(X_train,y_train)
        results.append(cv_results)
        names.append(name)
        dict[name] = [-cv_results.mean(), cv_results.std(), cv_results]
        msg = "%s: %f (%f)  " % (name, -cv_results.mean(), cv_results.std()), EmailFormatUtility.evaluate(model,X_test,y_test)
        print(msg)
    return dict


res = apply_loocv(X_train,y_train,X_test,y_test)


df.Server.value_counts()[:25]
#
#ds.to_csv("data/VirtualOneServer.csv")
#ds1.to_csv("data/VirtualFormattedonseXerver.csv")
#
#df.to_csv("data/virtualMemoryData.csv")

#plt.plot(df[["Time","MemoryUsed"]])
#plt.show()

from pandas.plotting import autocorrelation_plot
from matplotlib.pyplot import figure


figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')

autocorrelation_plot(ds["MemoryUsed"])
plt.show()


figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')

plt.plot(y_test)
plt.show()




rmse_val = []
for K in range(100):
    K = K+1
    model =KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = np.sqrt(mse(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

#StandardScaler, MinMaxScaler

#OneClassSVM , KNeighborsRegressor(n_neighbors = 7),SVR(kernel='linear')
#DecisionTreeRegressor, ExtraTreesRegressor
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', ExtraTreesRegressor())])
# prepare the model with target scaling
model_scaled = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
# evaluate model

model_scaled.fit(X_train,y_train)
NewRes = pd.DataFrame(index=y_test.index.copy())
NewRes["res"] =  model_scaled.predict(X_test)

NewRes["Actual"] = pd.DataFrame(y_test)["MemoryUsed"]
print(mae(y_test,NewRes["res"] ))
print(mse(y_test,NewRes["res"] ))
print(r2(y_test,NewRes["res"] ))

EmailFormatUtility.evaluate(model_scaled,X_test,y_test)


figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(pd.DataFrame(y_train), label='Train') 
plt.plot(pd.DataFrame(y_test), label='Valid') 
plt.plot(pd.DataFrame(NewRes["res"]), label='Prediction') 
plt.show()



feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


import statsmodels.api as sm
import statsmodels.tsa.api as smt

my_order = (1, 1, 1)
my_seasonal_order = (0, 1, 1, 7)


sm_model = sm.tsa.SARIMAX(X_train, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)

model_fit = sm_model.fit(disp=0)



plt.hist(y_train, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()