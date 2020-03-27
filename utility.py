# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:01:16 2020

@author: GJU5KOR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats as ss
from collections import Counter
import math
from sklearn.metrics import mean_absolute_error as mae

from nullHandler import remove_incomplete_samples,replace_nan_with_value

class EmailFormatUtility:
    
    @staticmethod
    def getPercentageFromUsed(val):
        val = str(val)
        ret = 0
        if(val and val.find('(')>-1):
            ret = val[val.find('(')+1 : len(val)-1]
            ret = ret.strip()
            if(ret.find('%')>-1):
                ret = ret.replace("%",'')
            if(ret.find(',')>-1):
                ret = ret.replace(',','.')
            ret= float(ret)
        return ret
    
    @staticmethod
    def getAllEmptyColumns(df):
        arycol = [x for x in df.columns if sum(df[x].isnull()) == df.shape[0]]
        return arycol
    
    @staticmethod
    def loadDiskDataset():
        df = pd.read_csv("data/DiskEmails.csv", parse_dates=["Time of Event"])
        df = df[df["Alert Classification"]=="disk"]
        df.drop(columns= EmailFormatUtility.getAllEmptyColumns(df),inplace=True)
        df.reset_index()
        return df
    
    @staticmethod
    def loadAllRecordsDataset():
        df = pd.read_excel("data/NetIQNew.xlsx", parse_dates=["Time of Event","CreationTime"])
        df.drop(columns= EmailFormatUtility.getAllEmptyColumns(df),inplace=True)
        df.reset_index()
        return df
    
    @staticmethod   
    def changeDataSizeCol(val):
        if(val and not pd.isna(val)):
            val = str(val)
            if(val.find("MB")>-1):
                val = val.replace("MB",'')
            if(val.find('(')>-1):
                val = val[0:val.find('(')]
            val = int(val)
            return val
        
    @staticmethod
    def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator/denominator)
        return eta
    @staticmethod
    def conditional_entropy_python(X, Y):
        """ 
        Calculate conditional entropy of all columns of X against Y (i.e. \sum_i=1^{N} H(X_i | Y)).
        """
        # Calculate distribution of y    
        Y_dist = np.zeros(shape=(int(Y.max()) + 1, ), dtype=np.float32)
        for y in range(Y.max() + 1):
            Y_dist[y] = (float(len(np.where(Y==y)[0]))/len(Y))
            
        Y_max = Y.max()
        X_max = X.max()
        
        ce_sum = 0.
        for i in range(X.shape[1]):
            ce_sum_partial = 0.
            
            # Count 
            counts = np.zeros(shape=(X_max + 1, Y_max + 1), dtype=np.int32)
            for row, x in enumerate(X[:, i]):
                counts[x, Y[row]] += 1
            
            # For each value of y add conditional probability
            for y in range(Y.max() + 1):
                count_sum = float(counts[:, y].sum())
                probs = counts[:, y] / count_sum
                entropy = -probs * np.log2(probs)
                ce_sum_partial += (entropy * Y_dist[y]).sum()
    
            ce_sum += ce_sum_partial
            
        return ce_sum
    
    @staticmethod
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x,y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    
    @staticmethod
    def theils_u(x, y):
        s_xy = EmailFormatUtility.conditional_entropy(x,y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
        s_x = ss.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x
        
    REPLACE = 'replace'
    DROP = 'drop'
    DROP_SAMPLES = 'drop_samples'
    DROP_FEATURES = 'drop_features'
    SKIP = 'skip'
    DEFAULT_REPLACE_VALUE = 0.0


    def conditional_entropy(x,
                            y,
                            nan_strategy=REPLACE,
                            nan_replace_value=DEFAULT_REPLACE_VALUE):
        """
        Calculates the conditional entropy of x given y: S(x|y)
        Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
        **Returns:** float
        Parameters
        ----------
        x : list / NumPy ndarray / Pandas Series
            A sequence of measurements
        y : list / NumPy ndarray / Pandas Series
            A sequence of measurements
        nan_strategy : string, default = 'replace'
            How to handle missing values: can be either 'drop' to remove samples
            with missing values, or 'replace' to replace all missing values with
            the nan_replace_value. Missing values are None and np.nan.
        nan_replace_value : any, default = 0.0
            The value used to replace missing values with. Only applicable when
            nan_strategy is set to 'replace'.
        """
        
        if nan_strategy == "replace":
            x, y = replace_nan_with_value(x, y, nan_replace_value)
        elif nan_strategy == DROP:
            x, y = remove_incomplete_samples(x, y)
        
        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x, y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0.0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y / p_xy)
        return entropy
    
    @staticmethod
    def getRecorswithMiminumCount(df,count=30):
        print("minimm value ",count)
        uniqServers = df.groupby(["JobID"])["Disk"].value_counts()
        reqJobs = []
        for(key, val) in uniqServers.iteritems():
            if(val>=count):
                print(val)
                reqJobs.append(key)
        return reqJobs
    
    
    @staticmethod
    def filterRowsData(row, recordsMinData):
            for v in recordsMinData:
                keys = str(v).split(" ")
                jobid = keys[0].replace('(','')
                jobid = jobid.replace(',','')
                jobid = jobid.replace('\'','')
                disk = keys[1].replace(')','')
                disk = disk.replace('\'','')
                if(int(row["JobID"]) == int(jobid) and str(row["Disk"].strip()) == disk):
                    return True
            return False
#        for v in recordsMinData:
#            keys = str(v).split(" ")
#            jobid = keys[0].replace('(','')
#            jobid = jobid.replace(',','')
#            jobid = jobid.replace('\'','')
#            disk = keys[1].replace(')','')
#            disk = disk.replace('\'','')
#        if(int(row["JobID"]) == int(jobid) and str(row["Disk"].strip()) == disk):
#            return True
#        return False
    
    
    
    
    @staticmethod
    def generateHeatMap(df,filterColumns):
        corrmat = df[filterColumns].corr()
        mask = np.zeros_like(corrmat, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots()
        fig.set_size_inches(10,10)
        sns.heatmap(corrmat,annot=True,mask=mask)
        
    @staticmethod
    def formatInitialData(df):
        aryDeleteCols = ["Server","Object Name","Severity","KS Name","EventID","Severity","Alert Classification","Type","Machine Location","KS Name","Source Machine Name","ErrorDetails","Detail Message",'Severity']
#        ,"ServerName"
        df["PercentageUsed"] = df["Currently used"].map(EmailFormatUtility.getPercentageFromUsed)
        df["Currently used"] = df["Currently used"].map(EmailFormatUtility.changeDataSizeCol)
        df["Total "] = df["Total "].map(EmailFormatUtility.changeDataSizeCol)  
        df["Currently free"] = df["Currently free"].map(EmailFormatUtility.changeDataSizeCol)
        df["Disk"] = df["Object Name"].map(lambda x: x.strip())
        df.drop(columns=aryDeleteCols,inplace=True)
        df.rename(columns={"Total ":"Total","Time of Event":'Time',"Event Severity":"Severity","Currently free":"free","Currently used":"used"},inplace=True)
        df["JobID"] = df["JobID"].map(lambda x:  str(int(x)) if not pd.isna(x) else 0 )
        df.reset_index(inplace=True)
        df.drop(columns=['used','index','free',"Severity","Total"],inplace=True)
        df = df.drop_duplicates()
        return df
    
    @staticmethod
    def extractInfoFromDates(df):
        df['isWeekEnd'] = df.Time.dt.dayofweek.apply(lambda x: 1 if x in [5,6] else 0)
        df["Hour"] = df.Time.dt.hour
        df['dayofweek'] = df.Time.dt.dayofweek
    #    df["Date"] = df.Time.dt.strftime("%Y-%m-%d")
        df["day"] = df.Time.dt.day
        df["month"] = df.Time.dt.month
#        df["year"] = df.Time.dt.year
        df['quarter'] = df.Time.dt.quarter
#        df['is_month_start'] = df.Time.dt.is_month_start
#        df['is_month_end'] = df.Time.dt.is_month_end
#        df['is_quarter_start'] = df.Time.dt.is_quarter_start
#        df['is_quarter_end'] = df.Time.dt.is_quarter_end
        return df
    
    @staticmethod
    def evaluate(model, test_features, test_labels):
      predictions = model.predict(test_features)
      errors = abs(predictions - test_labels)
      mape = 100 * np.mean(errors / test_labels)
      accuracy = 100 - mape
      msg = "Mean Abolute error ", mae(test_labels,predictions) , "Accuracy = {:0.2f}%.".format(accuracy)
      print(msg)
      return msg
  
  
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_pred - y_true) / y_pred)) * 100

    @staticmethod
    def getonlyMemoryUsed(x):
        if(x):
            if(str(x).find('>') >-1):
                x = x.split('>')[0]
            if(str(x).find(':') >-1):
                x = x.split(':')[1]
            if(str(x).find(',')>-1):
                x = x.replace(',','.')
        x = float(x)
        return x
    