# -*- coding: utf-8 -*-
"""
Created on Thu May  6 21:46:33 2021

@author: sakma
"""

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler


# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE

# Bagging
from sklearn.neighbors import KNeighborsClassifier

# Naive bayes
from sklearn.naive_bayes import GaussianNB


# Library imports
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


#LOADING DATASET
conflicting_data = pd.read_csv("C:\\Users\\sakma\\Desktop\\Ders Kayıtları\\ML-Dataset\\clinvar_conflicting.csv")

print(conflicting_data.shape)


#If there is such values consider them as NaN
conflicting_data = conflicting_data.replace(['-','not_specified','NULL'],np.nan)


#Percentage of null values for each feature
print("** Percentage of null values for each feature **\n")
total_num_of_values = conflicting_data.shape[0]
print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )


print("Before drop nan cols",conflicting_data.shape)

#If there is a column which has more than %60 percentage null we'll drop it.
conflicting_data.dropna(thresh=total_num_of_values*0.6,how='all',axis=1,inplace=True)
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )

print("After drop nan cols",conflicting_data.shape,"\n")



#summarize the content
# conflicting_data.info()


#We can get the count of unique values for each column.
for col in conflicting_data.columns:
    print(col+' '+str(len(conflicting_data[col].unique())))
    #print(conflicting_data[col].unique())
    print()



#CSDN OLANLAR İÇİN OBJECT -> INT YAPMALIYIZ.
corrmat = conflicting_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
plt.show()


#Correlation between CAD_RAW and CADD_PHRED is 0.96 then we'll drop one of them.
conflicting_data.drop(["CADD_RAW"], axis=1, inplace=True)























