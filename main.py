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

import time

start_time = time.time()

warnings.filterwarnings("ignore")

# LOADING DATASET
conflicting_data = pd.read_csv("C:\\Users\\sakma\\Desktop\\Ders Kayıtları\\ML-Dataset\\clinvar_conflicting.csv")

print(conflicting_data.shape)

# ********PREPROCESSING********#


print("Before drop nan cols", conflicting_data.shape)

# If there is a column which has more than %99 percentage(fully nan) null we'll drop it.
conflicting_data = conflicting_data.loc[:, conflicting_data.isnull().mean() < .99]
# print(((conflicting_data.isnull().sum()) / total_num_of_values ) * 100 )


''' 
In cDNA_position, CDS_position, Protein_position values are int but type of features are Object. 
Because there is some approximation such as 1331-1332 for relative positions. 
We'll take the valid one because they are already so close to each others
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
In CHROM column there are values between 1-22 and some X and MT values. X means chromosome X and MT means mitochondrial chromosome.
There is no Y chromosome therefore we can assume that our patient is a woman. We'll keep X as 23 and MT as 24. Then convert it's type into numeric
'''

conflicting_data['CHROM'] = conflicting_data['CHROM'].replace('X', 23)
conflicting_data['CHROM'] = conflicting_data['CHROM'].replace('MT', 24)
# It should be string because CHROM is a categorical column
conflicting_data[['CHROM']] = conflicting_data[['CHROM']].astype(str)

# correlation matrix for data
# corrmat = conflicting_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
# plt.show()

'''After analyzing correlation matrix we clearly see that 'cDNA_position', 'CDS_position','Protein_position'
features has 1 correlation between them. We can drop two of them and use just cDNA_position because it has less
nan value.
'''
conflicting_data.drop(['CDS_position', 'Protein_position'], axis=1, inplace=True)

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

'''
'Feature' is the ID of 'SYMBOL' therefore we'll drop the 'Feature' column and use the SYMBOL.
'''
conflicting_data.drop(['Feature'], axis=1, inplace=True)

'''
In 'POS','CLNHGVS', and 'CLNVI'(%53 NaN already and have 27k unique values) features are unique for each variant. Therefore we can drop them.
'''
conflicting_data.drop(['POS', 'CLNHGVS', 'CLNVI'], axis=1, inplace=True)

'''
'CLNVC' is stand for varient type. %94 of values are 'single_nucleotide' and the 6% of the values are
Deletion, Duplication, Inversion, or Insertion. We'll convert this column
as a binary column whether it is single base-pair substitution or not.
https://www.ebi.ac.uk/training/online/courses/human-genetic-variation-introduction/what-is-genetic-variation/types-of-genetic-variation/
At the same time we can conclude if the varient type is single base-pair or not from checking the
length of ALT and REF alleles. If one of the allele includes more than one base-pairs it means that
the varient type is multi base-pair substitution. Therefore we can drop the ALT and REF columns.

Same situation for ORIGIN
'''
# We can check the above situation
# print(conflicting_data[["ALT","REF","CLNVC"]][conflicting_data["CLNVC"]==1])


conflicting_data["CLNVC"] = np.where(conflicting_data["CLNVC"].str.contains("single_nucleotide"), 1, 0)
# if conflicting_data["ORIGIN"] == 1, assign 1, else assign 0.
conflicting_data["ORIGIN"] = np.where((conflicting_data["ORIGIN"] == 1), 1, 0)
conflicting_data.drop(['ALT', 'REF'], axis=1, inplace=True)

'''
ALT(Alternate allele) and Allele are the same. Then we can drop Allele because it has some null values but ALT doesn't
'''
conflicting_data.drop(['Allele'], axis=1, inplace=True)

'''
'MC' feature is the comma-separated list of molecular consequence in the form of Sequence Ontology and
'Consequence' feature is the Type of molecular consequence therefore we'll just keep the type of molecular
consequence and drop the 'MC' feature.
'''
conflicting_data.drop(['MC'], axis=1, inplace=True)

'''
'CLNDISDB' feature is the Tag-value pairs of disease database name and identifier, and
'CLNDN' feature is the disease name for the identifier. Therefore we'll just keep the
disease name and drop the 'CLNDISDB'
'''
conflicting_data.drop(['CLNDISDB'], axis=1, inplace=True)

# In CLNDN there is not_specified and not_provided seperately. We'll assume not_provided as not_specified.
conflicting_data['CLNDN'] = conflicting_data['CLNDN'].replace('not_provided', 'not_specified')

# We'll make CLNDN feature as binary. Whether Tag-value pairs have disease or not. If it has disease 1, else 0.
conflicting_data["pairs_has_disease"] = np.where((conflicting_data["CLNDN"] == 'not_specified'), 0, 1)
conflicting_data.drop(['CLNDN'], axis=1, inplace=True)

# Counts of unique values in feature
# print(conflicting_data['REF'].value_counts())
# print("**************************")
# print(conflicting_data['AF_ESP'].value_counts())


print("After drop nan cols", conflicting_data.shape, "\n")

##DEALING WITH MISSING VALUES##
# print(conflicting_data[["SYMBOL"]][conflicting_data["SYMBOL"].isnull()])
'''
In SYMBOL there is a missing gene. We'll call this gene X.
'''
conflicting_data['SYMBOL'] = conflicting_data['SYMBOL'].replace(np.nan, 'X')

# cDNA
'''
cDNA can be described as DNA without all the necessary noncoding regions. That's mean
if there is no EXON we can't talk about the cDNA position. There is an exception, lack of 
EXON and INTRON we have cDNA position. Therefore we'll fill the NaN values as -1
'''
conflicting_data['cDNA_position'] = conflicting_data['cDNA_position'].replace(np.nan, -1)

'''
CADD is a tool for scoring the deleteriousness of single nucleotide variants as well as insertion/deletions variants in the human genome.
All the NaN values in 'CADD_PHRED' belongs to non-single nucleotide varients. That's mean there is no CADD_PHRED information
for non-single nucleotide varient. We'll fill the values with -1.
'''
# print(conflicting_data[["CADD_PHRED","CLNVC"]][conflicting_data["CADD_PHRED"].isnull() & conflicting_data["CADD_PHRED"] == 1]) #it'll return empty array
conflicting_data['CADD_PHRED'] = conflicting_data['CADD_PHRED'].replace(np.nan, -1)
conflicting_data['CADD_RAW'] = conflicting_data['CADD_RAW'].replace(np.nan, -1)

'''
There are few NaN for Feature_type we'll keep them as unknown
'''
conflicting_data['Feature_type'] = conflicting_data['Feature_type'].replace(np.nan, 'unknown')

'''
Fill EXON NaN values with 0(it means there is no EXON)
'''
conflicting_data['EXON'] = conflicting_data['EXON'].replace(np.nan, 0)

'''
AminoAcids and Codons are only given if the varient affects protein-coding sequence.
'''
conflicting_data[['Amino_acids', 'Codons']] = conflicting_data[['Amino_acids', 'Codons']].replace(np.nan, "not-affect")

'''
Keep BAM_EDIT, SIFT, PolyPhen as unknown
'''
conflicting_data[['BAM_EDIT', 'SIFT', 'PolyPhen']] = conflicting_data[['BAM_EDIT', 'SIFT', 'PolyPhen']].replace(np.nan,
                                                                                                                "unknown")

'''
BLOSUM matrices are used to score alignments between evolutionarily divergent protein sequences.
BL0SUM62 is a categorical feature with integer values between -3 to 3. There is no 0. Fill NaN with 0
'''
conflicting_data['BLOSUM62'] = conflicting_data['BLOSUM62'].replace(np.nan, 0)

'''
the DNA STRAND (1 or -1) on which the transcript/feature lies. There are 14 NaN values. Keep them as 0
'''
conflicting_data['STRAND'] = conflicting_data['STRAND'].replace(np.nan, 0)

'''
Fill LoFtool with the mean of feature.
'''
# print("Mean of LoFtool",conflicting_data['LoFtool'].mean())
from sklearn.impute import SimpleImputer

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(conflicting_data[['LoFtool']])
conflicting_data['LoFtool'] = imr.transform(conflicting_data[['LoFtool']]).ravel()

# Now we don't have any NaN values.


# summarize the content
conflicting_data.info()

# correlation matrix for data
# corrmat = conflicting_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
# plt.show()


# ENCODING


# categorical_features = ['CHROM', 'IMPACT', 'STRAND', 'BAM_EDIT', 'SIFT', 'PolyPhen',
#                         'BLOSUM62','Consequence']

binary_cols = ['CLNVC', 'ORIGIN', 'pairs_has_disease']
nominal_cols = ['SIFT', 'BAM_EDIT', 'PolyPhen', 'BIOTYPE', 'Feature_type']
# nominal cols more than 20 unique values.
feature_hashing_cols = ['SYMBOL', 'Amino_acids', 'Codons']

# First of all IMPACT is ordinal column we'll mapping manuelly
impact_ord_map = {'LOW': 0, 'MODERATE': 1, 'MODIFIER': 2, 'HIGH': 3}
conflicting_data['IMPACT'] = conflicting_data['IMPACT'].map(impact_ord_map)

# We'll apply OneHotEncoding for nominal_cols with get_dummies. First encode str values into numeric values
for col in nominal_cols:
    conflicting_data = pd.concat([conflicting_data, pd.get_dummies(conflicting_data[col], prefix=col + '_')], axis=1)
    conflicting_data.drop([col], axis=1, inplace=True)


def generate_col_names(col, n_features):
    col_names = list()
    for i in range(n_features):
        col_names.append(col + "{}".format(i))
    return col_names


from sklearn.feature_extraction import FeatureHasher

# 24 unique values for CHROM
fh = FeatureHasher(n_features=4, input_type='string')
hashed_features = fh.fit_transform(conflicting_data['CHROM'])
hashed_features = hashed_features.toarray()
conflicting_data = pd.concat([conflicting_data, pd.DataFrame(hashed_features, columns=generate_col_names('CHROM', 4))],
                             axis=1)

# 48 unique values for Consequence
fh = FeatureHasher(n_features=16, input_type='string')
hashed_features = fh.fit_transform(conflicting_data['Consequence'])
hashed_features = hashed_features.toarray()
conflicting_data = pd.concat(
    [conflicting_data, pd.DataFrame(hashed_features, columns=generate_col_names('Consequence', 16))],
    axis=1)

conflicting_data.drop(['CHROM', 'Consequence'], axis=1, inplace=True)

# For feature_hashing_cols we have categorical values more than 1000. We'll apply FeatureHasher with n_features = 16
for col in feature_hashing_cols:
    n = 8
    fh = FeatureHasher(n_features=n, input_type='string')
    hashed_features = fh.fit_transform(conflicting_data[col])
    hashed_features = hashed_features.toarray()
    conflicting_data = pd.concat([conflicting_data, pd.DataFrame(hashed_features, columns=generate_col_names(col, n))],
                                 axis=1)
    conflicting_data.drop([col], axis=1, inplace=True)

# We can get the count of unique values for each column.
# for col in conflicting_data.columns:
#     print(col+' '+str(len(conflicting_data[col].unique())) +' ',(conflicting_data[col]).dtype)
#     if(col == "PolyPhen" or col == "IMPACT" or col == "SIFT" or col == "pairs_has_disease" ):
#         print(conflicting_data[col].unique())
#     print()


# There is natural order in IMPACT -> ['MODERATE' 'MODIFIER' 'LOW' 'HIGH']
# BAM_EDIT 3 [nan 'OK' 'FAILED'] %51 null %49 OK %1 FAILED
# PolyPhen 5 ['benign' 'probably_damaging' nan 'possibly_damaging' 'unknown']
# SIFT 5 ['tolerated' 'deleterious_low_confidence' 'deleterious' nan 'tolerated_low_confidence']


from sklearn.decomposition import PCA

y = conflicting_data.CLASS
X = conflicting_data.drop(["CLASS"], axis=1, inplace=False)

print("Before sampling:", X.shape, y.shape)

rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rnd_clf.fit(X, y)

features = X.columns
importances = rnd_clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(20, 10))
feat_importances = pd.Series(importances, index=features)
feat_importances.nlargest(len(indices)).plot(kind='bar', color='#79CCB3');
plt.show()

# we will use the implementations provided by the imbalanced-learn Python library, which can be installed via pip as follows:
# sudo pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

# RANDOMOVERSAMPLER %84 ACCURACY 97.5K DATA
# SMOTE %78 ACCURACY 97.5K DATA
# ADASYN %78.9 ACCURACY WITH 99K DATA
# ros = ADASYN(random_state=0)
# pca = PCA(n_components=15)
# X = pca.fit_transform(X)
# ros = RandomUnderSampler(random_state=0,sampling_strategy=0.8)
# X_resampled, y_resampled = ros.fit_resample(X, y)

# y = y.to_frame()
# ax = sns.countplot(x="CLASS", data=y)
# ax.set(xlabel='CLASS', ylabel='Number of Variants')
# plt.show()

# y_resampled = y_resampled.to_frame()
# ax = sns.countplot(x="CLASS", data=y_resampled)
# ax.set(xlabel='CLASS', ylabel='Number of Variants')
# plt.show()

# print("After sampling:",X_resampled.shape, y_resampled.shape)


# We split the data into train(%75) and test(%25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ros = RandomUnderSampler(random_state=0, sampling_strategy=0.8)
X_train, y_train = ros.fit_resample(X_train, y_train)

y_train_raw = y_train.to_frame()
ax = sns.countplot(x="CLASS", data=y_train_raw)
ax.set(xlabel='CLASS', ylabel='Number of Variants')
plt.title("y_train")
plt.show()

y_test_raw = y_test.to_frame()
ax = sns.countplot(x="CLASS", data=y_test_raw)
ax.set(xlabel='CLASS', ylabel='Number of Variants', label="y_test")
plt.title("y_test")
plt.show()

# print("After sampling:",X_resampled.shape, y_resampled.shape)
from sklearn.metrics import f1_score

machine_learning_algorithms = (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                          max_depth=7, random_state=0),
                               LogisticRegression(solver='liblinear'),
                               RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=50,
                                                      max_features="auto", min_samples_leaf=50),
                               DecisionTreeClassifier(max_depth=7),
                               )
ml_names = ("GradientBoost", "Logistic Regression", "RandomForest", "DecisionTree")

for ml, ml_name in zip(machine_learning_algorithms, ml_names):
    clf = ml
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    # print("{} Accuracy: %".format("SVC"), 100 - mean_absolute_error(y_test, predict) * 100)
    print("{} Accuracy: %".format("Accuracy score:"), accuracy_score(y_test, predict) * 100)
    mae = mean_absolute_error(y_test, predict)
    print('MAE: %.3f' % mae)
    print("{} Accuracy: %".format("ROC"), roc_auc_score(y_test, predict) * 100)
    print("Classification Report : for:", ml_name, "\n", classification_report(y_test, predict))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # roc_auc = roc_auc_score(y_test, predict)
    print("AUC:", roc_auc)
    fpr, tpr, _ = roc_curve(y_test, predict)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, label='{} AUC = {:.2f}'.format(ml_name, roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    print("*********************")

# clf = GradientBoostingClassifier()
# clf.fit(X_train, y_train)
# predict = clf.predict(X_test)
# print(X_train.shape, y_train.shape)
# print("{} Accuracy: %".format("SVC"), 100 - mean_absolute_error(y_test, predict) * 100)
# print("{} Accuracy: %".format("AUC"), roc_auc_score(y_test, predict) * 100)
# print('Average precision-recall score: {0:0.2f}'.format(average_precision_score(y_test, predict)))
# print( "Classification Report :\n ", classification_report(y_test, predict))

# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_recall_fscore_support as score
# precision_lr, recall_lr = (round(float(x),2) for x in list(score(y_test,
#                                                                     predict,
#                                                                     average='weighted'))[:-2])


# rfe = RFE(clf,15)
# rfe.fit(X_train, y_train)
# predict = rfe.predict(X_test)
# print("RFE1 Accuracy: %", 100 - mean_absolute_error(y_test, predict) * 100)


####UNDERSAMPLING YAPTIM SADECE TRAIN DATASI IÇIN ORDAN DEVAMKE


print("--- %s seconds ---" % (time.time() - start_time))
