#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import xgboost as xgb
import statsmodels.api as sm
from pylab import *
from matplotlib import style
from heapq import nlargest
from pandas.plotting import register_matplotlib_converters,scatter_matrix
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor,TPOTClassifier
from xgboost.sklearn import XGBRegressor

pd.options.mode.chained_assignment = None 

def get_headers(df):
    return(df.columns.values)

def read_data(path,methd):
    methods = ['ffill','bfill','mean','zero']
    try:
        if(methd == 'ffill'):
            df = pd.read_csv(path,sep=',')
            df.fillna(method='ffill',inplace=True)
            return(df)
        elif(methd == 'bfill'):
            df = pd.read_csv(path,sep=',')
            df.fillna(method='bfill',inplace=True)
            return(df)
        elif(methd == 'mean'):
            df = pd.read_csv(path,sep=',')
            df.fillna(df.mean(),inplace=True)
            return(df)
        elif(methd == 'zero'):
            df = pd.read_csv(path,sep=',')
            df.fillna(0,inplace=True)
            return(df)
        elif(methd not in methods):
            raise NameError
    except NameError:  print ('Not correct filling method')   
    
def view_data(df):
    print(df.head(20))

def get_info(df):  
    df.info()
    df.describe()
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
   
def check_lin(df):
    cols = [col for col in df.columns if 'y' not in col  ]
    for el in cols:
        fig = plt.figure() 
        plt.scatter(df[str(el)], df['y'], color='red')
        plt.xlabel(str(el), fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        fig.savefig("figs/y_vs_"+str(el), bbox_inches='tight')
        plt.close()
      
def check_features(df,flag):
    cols = [ col for col in df.columns if str(flag) in col ] + ['y']
    scatter_matrix(df[cols], figsize=(15, 10), diagonal='kde')
    plt.show()

def get_cmtx_feat(df,corr_thrs):
    corr_matrix = df.corr()
    cmtx = corr_matrix["y"].sort_values(ascending=False)    
    dct = dict(zip(list(cmtx.keys()),cmtx.values))
    dct_select = dict((k, v) for k, v in dct.items() if v >= float(corr_thrs)/100.0 and k != 'y')
    return(dct_select)

def plot_focus(df,top_features):
    df.plot(kind="scatter", x=str(top_features[1]), y="y", alpha=0.5)
    plt.show()
  
def ml_base(df,top_features,check_shape,regrs):
    df = df[top_features + ['y']]
    X = np.array(df.drop(['y'],axis=1))
    X = preprocessing.scale(X)
    y = np.array(df['y'])  
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)    

    if(check_shape):
        print 'X_train.shape, X_test.shape = ', \
            X_train.shape, X_test.shape
        print 'y_train.shape, y_test.shape = ', \
            y_train.shape, y_test.shape

    if(regrs == 'lin'):
        reg = LinearRegression()
    elif(regrs == 'svm-lin'):
        reg = svm.SVR(kernel='linear',gamma='auto')
    elif(regrs == 'svm-poly'):
        reg = svm.SVR(kernel='poly',gamma='auto')
    elif(regrs == 'tree'):
        reg = DecisionTreeRegressor(random_state=23)
    elif(regrs == 'forest'):
        reg = RandomForestRegressor(n_estimators=100,random_state=23)
    elif(regrs == 'xgbr'):
        reg = XGBRegressor(learning_rate=1.0, max_depth=5, min_child_weight=15, \
                           n_estimators=100, nthread=1, subsample=0.6)
    else:
        print('Input acorrect classifer choice')

    reg.fit(X_train,y_train)
    predictions = reg.predict(X_test)
    accuracy = reg.score(X_test,y_test)
    display_output(accuracy,predictions,y_test)

    # plot_ml(y_test,predictions)
    
def plot_ml(test,pred):
    plt.plot(test,'r*-')
    plt.plot(pred,'bo-')
    plt.xlabel('y - true values')
    plt.ylabel('y - predictions')
    plt.show()      

def display_scores(scores):
    print('###########################################################')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('###########################################################')

def display_output(acc,pred,act_vals):
    print('###########################################################')
    print 'accuracy    = ', acc
    print 'y predictions = ', map(lambda x: '{0:.2f}'.format(x),list(pred))
    print 'actual vals = ', list(act_vals)         
    print('###########################################################')
         
def ml_tpot(df,top_classes):
    # tpot allows to run different pipelines and output a model
    # which can them be implemented (this is like an automatic ML selection model)
    df = df[top_classes + ['y']]
    X = np.array(df.drop(['y'],axis=1))
    X = preprocessing.scale(X)
    y = np.array(df['y'])  

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)    
 
    tpot = TPOTRegressor(generations=10,verbosity=2,scoring='r2',n_jobs=-1,random_state=23)
    tpot.fit(X_train, y_train)
    predictions = tpot.predict(X_test)
    accuracy = tpot.score(X_test,y_test)
    tpot.export('model_export.py')
    display_output(accuracy,predictions,y_test)
    
if __name__ == '__main__':
    dirc = os.path.join(os.getcwd(),'data.csv')

    df_raw = read_data(dirc,'zero')

    # (1) plot features scatterplot
    flags = [ 'bidRate', 'bidSize', 'askRate', 'askSize' ]
    index = 0
    check_features(df_raw,flags[index])

    # (2) correlation matrix
    corr_thrs = 15 
    cm = get_cmtx_feat(df_raw,corr_thrs)
    # get top features after threshold selection
    top_features = nlargest(len(cm),cm,key=cm.get)
    
    # (3) basic ml regressors
    regrs = [ 'lin', 'svm-lin', 'svm-poly', 'tree', 'forest' ]
    idx = 4
    ml_base(df_raw,top_features,False,regrs[idx])
   
    # (4) test auto ML (using Tpot)
    ml_tpot(df_raw,top_features)

    
