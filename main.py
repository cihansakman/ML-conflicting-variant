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

# ********PREPROCESSING********#


# If there is such values consider them as NaN
conflicting_data = conflicting_data.replace(['-', 'not_specified', 'NULL'], np.nan)

# Percentage of null values for each feature
# print("** Percentage of null values for each feature **\n")
# total_num_of_values = conflicting_data.shape[0]
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )


print("Before drop nan cols", conflicting_data.shape)

# If there is a column which has more than %99 percentage(fully nan) null we'll drop it.
conflicting_data = conflicting_data.loc[:, conflicting_data.isnull().mean() < .99]
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )


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

'''
In CHROM collum there is values between 1-22 and some X and MT values. X means chromosome X and MT means mitochondrial chromosome.
There is no Y chromosome therefore we can assume that our patient is a woman. We'll keep X as 23 and MT as 24. Then convert it's type into numeric
'''

conflicting_data['CHROM'] = conflicting_data['CHROM'].replace('X', 23)
conflicting_data['CHROM'] = conflicting_data['CHROM'].replace('MT', 24)
conflicting_data[['CHROM']] = conflicting_data[['CHROM']].apply(pd.to_numeric)

# correlation matrix for data
corrmat = conflicting_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
plt.show()

'''After analyzing correlation matrix we clearly see that 'cDNA_position', 'CDS_position','Protein_position'
features has 1 correlation between them. We can drop two of them and use just cDNA_position because it has less
nan value.
'''
conflicting_data.drop(['CDS_position', 'Protein_position'], axis=1, inplace=True)

# Correlation between CAD_RAW and CADD_PHRED is 0.96 then we'll drop one of them.
conflicting_data.drop(["CADD_RAW"], axis=1, inplace=True)

'''
For EXON and INTRON their NaN ratios are parallel. The percentage of NaN values for INTRON is %86.4,
and the percentage of NaN values for EXON is %13.6. Therefore we'll fill the NaN values of EXON with
the corresponding values of INTRON
'''
conflicting_data["EXON"][conflicting_data["EXON"].isnull()] = conflicting_data["INTRON"][
    conflicting_data["INTRON"].notnull()]
# Now we can drop the INTRON
conflicting_data.drop(["INTRON"], axis=1, inplace=True)
# print(conflicting_data[["EXON","INTRON"]][conflicting_data["INTRON"].notnull() & conflicting_data["EXON"].isnull()])

'''
EXON feature is stand for the exon number (out of total number). That's mean we can covert it into float.
'''


# Here we have a function which will make EXON values float
def makeExonFloat(df, colname):
    index = 0;
    # nan values' type is float, others str
    for value in df[[colname]].values:
        # If the value is in form a/b(etc.) there will be a ValueError when try to find math.isnan.
        try:
            math.isnan(value)

        except ValueError:
            split_form = value[0].split("/")
            df[colname][index] = float(split_form[0]) / float(split_form[1])

        index += 1
    return df


# call the function
conflicting_data = makeExonFloat(conflicting_data, 'EXON')
# convert EXON column to float
conflicting_data[['EXON']] = conflicting_data[['EXON']].apply(pd.to_numeric)

# #summarize the content
# conflicting_data.info()


'''
In 'POS','CLNHGVS', and 'CLNVI'(%53 NaN already and have 27k unique values) features are unique for each variant. Therefore we can drop them.
'''
conflicting_data.drop(['POS', 'CLNHGVS', 'CLNVI'], axis=1, inplace=True)

'''
'Feature_type' ,'BIOTYPE' all values are same
'''

print("After drop nan cols", conflicting_data.shape, "\n")

print("**************************")

# We can get the count of unique values for each column.
for col in conflicting_data.columns:
    print(col + ' ' + str(len(conflicting_data[col].unique())) + ' ', (conflicting_data[col]).dtype)
    if (
            col == "BIOTYPE" or col == "CLNVC" or col == "STRAND" or col == "BAM_EDIT" or col == "SIFT" or col == "PolyPhen" or col == "BLOSUM62"):
        print(conflicting_data[col].unique())
    print()

# There is natural order in IMPACT -> ['MODERATE' 'MODIFIER' 'LOW' 'HIGH']
# BAM_EDIT 3 [nan 'OK' 'FAILED']
# PolyPhen 5 ['benign' 'probably_damaging' nan 'possibly_damaging' 'unknown']
# SIFT 5 ['tolerated' 'deleterious_low_confidence' 'deleterious' nan 'tolerated_low_confidence']

