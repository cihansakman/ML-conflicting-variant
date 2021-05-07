# -*- coding: utf-8 -*-
"""
Created on Thu May  6 21:46:33 2021

@author: sakma
"""

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint
import math

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

# LOADING DATASET
conflicting_data = pd.read_csv("C:\\Users\\sakma\\Desktop\\Ders Kayıtları\\ML-Dataset\\clinvar_conflicting.csv")

print(conflicting_data.shape)

# If there is such values consider them as NaN
conflicting_data = conflicting_data.replace(['-', 'not_specified', 'NULL'], np.nan)

# Percentage of null values for each feature
print("** Percentage of null values for each feature **\n")
total_num_of_values = conflicting_data.shape[0]
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )


print("Before drop nan cols", conflicting_data.shape)

# If there is a column which has more than %60 percentage null we'll drop it.
conflicting_data.dropna(thresh=total_num_of_values * 0.6, how='all', axis=1, inplace=True)
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )

print("After drop nan cols", conflicting_data.shape, "\n")

# summarize the content
# conflicting_data.info()


# #We can get the count of unique values for each column.
# for col in conflicting_data.columns:
#     print(col+' '+str(len(conflicting_data[col].unique())))
#     #print(conflicting_data[col].unique())
#     print()


''' 
In cDNA_position, CDS_position, Protein_position values are int but type of features are Object. 
Because there is some approximation such as 1331-1332 for relative positions. 
We'll take the first one because they are already so close to each others
'''


# Here we have a function which will make the update
def splitValues(df, colname):
    index = 0;
    # nan values' type is float, others str
    for value in df[[colname]].values:
        # If the value is in form 123-124(etc.) there will be a ValueError when try to find math.isnan.
        try:
            math.isnan(value)

        except ValueError:
            split_form = value[0].split("-")
            if (split_form[0] != '?'):
                df[colname][index] = split_form[0]
            else:
                df[colname][index] = split_form[1]

        index += 1
    return df


# Convert all values into float.
for i in ['cDNA_position', 'CDS_position', 'Protein_position']:
    conflicting_data = splitValues(conflicting_data, i)

# Now we shold convert our columns' type into numeric.
conflicting_data[['cDNA_position', 'CDS_position', 'Protein_position']] = conflicting_data[
    ['cDNA_position', 'CDS_position', 'Protein_position']].apply(pd.to_numeric)

# CHROM there is some X values we'll assume that these are NaN values.


# CSDN OLANLAR İÇİN OBJECT -> INT YAPMALIYIZ.
corrmat = conflicting_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
plt.show()

# Correlation between CAD_RAW and CADD_PHRED is 0.96 then we'll drop one of them.
conflicting_data.drop(["CADD_RAW"], axis=1, inplace=True)

# summarize the content
conflicting_data.info()


















