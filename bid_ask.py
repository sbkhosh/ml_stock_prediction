#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xgboost as xgb
import seaborn
import warnings
from xgboost.sklearn import XGBRegressor
from pylab import *
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
    RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.metrics import make_scorer, r2_score

from pandas.plotting import register_matplotlib_converters
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

SCALE_INDEX = 0
FILL_INDEX = 3

def get_headers(df):
    return(list(df.columns.values))

def read_data(path,index):
    fillers = def_fill()
    methods = fillers.values()

    if(isinstance(index, (int))):
        methd = fillers[index]
    else:
        methd = ''
    
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
            df.fillna(0.0,inplace=True)
            return(df)       
        elif(methd not in methods):
            raise NameError
    except NameError:
        # print ('Not correct filling method')
        df = pd.read_csv(path,sep=',')
        # return by default the raw data (with missing data)
        return(df)
   
def view_data(df):
    print(df.head(20))
  
def plot_ts(ts):
    plt.plot(ts)
    plt.show()

def ratio_missing(df):    
    ser = pd.Series.to_frame(df.isnull().sum()/len(df)*100)
    dct = pd.DataFrame.to_dict(ser)[0]
    res = [ (k,v) for k, v in dct.items() if v > 0.0 ]
    return(res)
    
def adf(v, crit='5%', max_d=6, reg='nc', autolag='AIC'):
    boolean = False    
    adf = adfuller(v, max_d, reg, autolag)
    if(adf[0] < adf[4][crit]):
        pass
    else:
        boolean = True
    return boolean

def get_adfs(df):
    headers = get_headers(df)
    adf_ts = [ (headers[el],adf(df[headers[el]])) for el in range(len(headers)) ]
    return(adf_ts)

def scaler_def(X,index):
    if(index == 0):
        return(StandardScaler().fit_transform(X))
    elif(index == 1):
        return(MinMaxScaler().fit_transform(X))
    elif(index == 2):
        return(MaxAbsScaler().fit_transform(X))
    elif(index == 3):
        return(RobustScaler(quantile_range=(25, 75)).fit_transform(X))
    elif(index == 4):
        return(PowerTransformer(method='yeo-johnson').fit_transform(X))
    elif(index == 5):
        return(PowerTransformer(method='box-cox').fit_transform(X))
    elif(index == 6):
        return(QuantileTransformer(output_distribution='normal').fit_transform(X))
    elif(index == 7):
        return(QuantileTransformer(output_distribution='uniform').fit_transform(X))
    elif(index == 8):
        return(Normalizer().fit_transform(X))
    else:
        print('not a correct scaler defined')

def get_pca(df,components,index,plotting):
    df = df.drop(['y'],axis=1)
    headers = get_headers(df)

    data_rescaled = scaler_def(df[headers].values,index)
    pca = PCA(n_components=components)
    pca_res = pca.fit_transform(data_rescaled)

    if(plotting):
        plot_pca(pca)

    return(pca,pca_res)
    
def plot_pca(pca):
    plt.figure(figsize=(15,8))
    plt.plot(range(components), pca.explained_variance_ratio_,'ro')
    plt.plot(range(components), np.cumsum(pca.explained_variance_ratio_),'b*')
    plt.title("Explained Variance")
    plt.axhline(y=0.98, color='m', linestyle='-')
    plt.legend(['Component-wise','Cumulative','98% threshold'])
    plt.show()

def train_test_data(df,pca_res,pca_feat):
    if(pca_feat):
        train = pca_res
    else:
        train = np.array(df.drop(['y'],axis=1))

    labels = np.array(df['y'])  

    results={}
    def test_model(clf):
        
        cv = KFold(n_splits=5,shuffle=True,random_state=45)
        r2 = make_scorer(r2_score)
        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=r2)
        scores=[r2_val_score.mean()]
        return scores

    clfs = [ linear_model.Ridge(),  linear_model.BayesianRidge(), linear_model.HuberRegressor(), linear_model.Lasso(alpha=1e-4), \
             BaggingRegressor(), RandomForestRegressor(), AdaBoostRegressor(), svm.SVR() ]
    res_cols = [ "Ridge", "Bayesian Ridge", "Hubber", "Lasso", "Bagging", "RandomForest", "AdaBoost", "SVM RBF" ]

    for clf, col in zip(clfs,res_cols):
        results[str(col)] = test_model(clf)
    return results

def train_test_data_plot(results):
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["R Square Score"] 
    results.sort_values(by=["R Square Score"],ascending=False)

    results.plot(kind="bar",title="Model Scores",figsize=(15,8))
    axes = plt.gca()
    axes.set_ylim([0,1])

    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.axhline(y=0.25, color='m', linestyle='--')
    plt.axhline(y=0.5, color='m', linestyle='--')
    plt.axhline(y=0.75, color='m', linestyle='--')
    plt.show()

def train_test_data_comb(results,results_pca,scaler_index,filler_index):
    res = pd.DataFrame.from_dict(results,orient='columns')
    res_pca = pd.DataFrame.from_dict(results_pca,orient='columns')
    res.index = ['without PCA']
    res_pca.index = ['with PCA']
    res_all = pd.concat([res,res_pca])

    scalers, fillers = def_scalers(), def_fill()
    scaler_nm = scalers[scaler_index]
    filler_nm = fillers[filler_index]
   
    res_all.plot(kind='bar',title='R2 model scores - ' + str(scaler_nm) + ' - ' + str(filler_nm),rot=0,figsize=(15,8))
    axes = plt.gca()
    axes.set_ylim([0,1])

    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.axhline(y=0.25, color='m', linestyle='--')
    plt.axhline(y=0.5, color='m', linestyle='--')
    plt.axhline(y=0.75, color='m', linestyle='--')
    plt.savefig('train_test_pca_compare_' + str(scaler_nm) + '_' + str(filler_nm) + '.pdf')
    # plt.show()

def def_scalers():
    dct_scalers = { 0: 'StandardScaler', 
                    1: 'MinMaxScaler'  ,
                    2: 'MaxAbsScaler'  ,
                    3: 'RobustScaler'  ,
                    4: 'PowerTransformer' ,
                    5: 'QuantileTransformer' ,
                    6: 'Normalizer' }   
    return(dct_scalers)

def def_fill():
    dct_fill = { 0: 'ffill',
                 1: 'bfill', 
                 2: 'mean' ,
                 3: 'zero' }
    return(dct_fill)
   
if __name__ == '__main__':
    # dirc = os.path.join(os.getcwd(),'data-training.csv')
    dirc = os.path.join(os.getcwd(),'data.csv')
  
    df_raw = read_data(dirc,'')
    df_fill = read_data(dirc,FILL_INDEX)
   
    # (1) missig values percentage over all dataset
    ratio_miss = ratio_missing(df_raw)
    
    # (2) adf computation for all features and label
    all_adfs = get_adfs(df_fill)
    
    # (3) pca of the features
    components, index = 20, SCALE_INDEX
    pca, pca_res = get_pca(df_fill,components,SCALE_INDEX,False)
    
    # (4) train/test data with all features
    trtstscore = train_test_data(df_fill,pca_res,False)
    train_test_data_plot(trtstscore)
    
    # (5) train/test data with PCA features
    trtstscore_pca = train_test_data(df_fill,pca_res,True)
    train_test_data_plot(trtstscore_pca)

    # (6) compare with and without PCA versions
    train_test_data_comb(trtstscore,trtstscore_pca,SCALE_INDEX,FILL_INDEX)





    
