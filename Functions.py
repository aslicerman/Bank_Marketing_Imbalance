#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import difflib
import cx_Oracle as cx
from tqdm import tqdm
import sqlalchemy

pd.set_option("display.max_columns",None);
pd.set_option("display.max_rows",None);
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap

from scipy import stats
from scipy.stats import norm, skew

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# In[ ]:


def target_analysis(target, df):
    palette = {0: "darkred", 
             1: (0.6, 0.6, 0.6)}

    print("Yes : ", len(df[df[target]=="yes"]),
          "Yes Ratio :", "{:.2%}".format(len(df[df[target]=="yes"])/len(df)))

    plt.bar(['No', 'Yes'], df.y.value_counts().values, facecolor = 'brown', edgecolor='brown', linewidth=0.5, ls='dashed')
    sns.set(font_scale=1)
    plt.title('Target Status', fontsize=14)
    plt.xlabel('Status')
    plt.ylabel('Number of Customer')
    plt.show()
        
def nunique(df, cols):
    return df[cols].nunique().sort_values(ascending = False)

def unique(df, cols):
    for i in cols:       
        print("{} : ".format(i))
        print(df[i].unique())
        print("""""")
        
def columns_dtypes(df):
    categorical_feats = df.dtypes[df.dtypes =="object"].index
    numerical_feats = df.dtypes[df.dtypes !="object"].index
    return categorical_feats, numerical_feats

def missing_values_table(df):
    total = df.isnull().sum()
    percent =df.isnull().sum()/len(df)
    missing_data = pd.concat([total[percent>0], percent[percent>0]], axis=1, keys=['Total', 'Percent']).sort_values(by = "Percent", ascending=False).round(2)
    print("Veri setinde", len(df.columns), "değişken mevcut.",  len(missing_data), "adet değişken eksik veriye sahip")
    return missing_data

def zero_values(df,numerical_feats):
    cols= []
    counts = []
    for j in range(0, len(numerical_feats)):
        cols.append(numerical_feats[j])
        counts.append(len(df[df[numerical_feats[j]]==0]))
    zero = pd.DataFrame({'cols': cols, 'counts': counts}).sort_values("counts",ascending=False)
    zero["per_zero"] = zero["counts"]/len(df)
    return zero

def one_hot_encoder(df, categorical_cols, nan_as_category=True):
    original_columns = list(df.columns)
    dataframe = pd.get_dummies(df, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    dataframe .columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dataframe .columns]
    new_columns = [c for c in df.columns if c not in original_columns]
    return dataframe, new_columns            


def stacked(df,col, target):    
    color = {"yes": "darkred", "no": "Grey"}
    data = df.groupby([col,target]).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
    ax =data.plot(kind='bar',stacked=True,width=0.8,figsize=[16,5],sort_columns=True,fontsize=12,alpha=1, 
                                                                                                             color = color) 
    #Set txt
    kx=-0.3
    ky=-0.02

    plt.xticks(x=df.index.name ,rotation=0)
    values =  list(1-data["yes"]/(data["yes"] +data["no"])) + list(data["yes"]/(data["yes"] +data["no"]) )

    for i,rec in enumerate(ax.patches):
        ax.text(rec.get_xy()[0]+rec.get_width()/1.5+kx,rec.get_xy()[1]+rec.get_height()/4+ky,'{:.1%}'.format(values[i]),fontsize=12,weight="bold", color='k')
    plt.show()
    
def bar_plot(df,col,target,values):
    data=df.sort_values(by=col, ascending=True)
    plt.figure(figsize=[16,5])
    sns.set(font_scale=1)
    plot=sns.countplot(x=col, hue=target, data=data,palette=["darkred", "darkgrey"])
    plt.xticks(x=col ,rotation=0)

    x=data.pivot_table(index=[col],columns=[target],values=values,aggfunc="count")

    for p, label in zip(plot.patches, x["yes"]):
        plot.annotate(label, (p.get_x()+0.5, p.get_height()/3), ha='left', va='bottom',rotation=0,weight="bold",color='k')
    for p, label in zip(plot.patches, x["no"]):
        plot.annotate(label, (p.get_x()+0.22, p.get_height()+0.15),ha='right',  va='bottom',rotation=0,weight="bold",color='k')

    plt.show()
    
def outlier_thresholds(df, feature):
    quantile1 = df[feature].quantile(0.1)
    quantile3 = df[feature].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def has_outliers(df, numerical_cols):   
    for col in numerical_cols:
        low_limit, up_limit = outlier_thresholds(df, col)
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            number_of_outliers = df[(df[col] > up_limit) | (df[col] < low_limit)].shape[0]      
            print(col, ":", number_of_outliers)

def outliers_update_up(df,updated_outliers):
    low, up = outlier_thresholds(df, updated_outliers)
    for col in df[updated_outliers]:
        low, up = outlier_thresholds(df, col)
        df[col] = df[col].apply(lambda x: up if (x>up) else x)
        
def distplot(df, target,num_list,nr_rows,nr_cols):

    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*8,nr_rows*4))
    for r in range(0,nr_rows):
        for c in range(0,nr_cols):  
            i = r*nr_cols+c
            if i < len(num_list):
                sns.distplot(df[df[target]=="yes"][num_list[i]],label='yes',ax = axs[r][c],color="#016795")
                sns.distplot(df[df[target]=="no"][num_list[i]],label='no',ax = axs[r][c],color="#920a4e")
                sns.set_style({"axes.facecolor": 'White'})
                sns.set_context("paper", font_scale=2)
                sns.set_style("ticks", {"xtick.major.size": 14, "ytick.major.size": 14})
                sns.axes_style({'xtick.color': '.12','ytick.color': '.12'})    
    plt.tight_layout()    
    plt.show()

