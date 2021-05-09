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
# corrmat = conflicting_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
# plt.show()

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

'''
'Feature' is the ID of 'SYMBOL' therefore we'll drop the 'Feature' column.
'''
conflicting_data.drop(['Feature'], axis=1, inplace=True)

'''
In 'POS','CLNHGVS', and 'CLNVI'(%53 NaN already and have 27k unique values) features are unique for each variant. Therefore we can drop them.
'''
conflicting_data.drop(['POS', 'CLNHGVS', 'CLNVI'], axis=1, inplace=True)

# ***IMBALANCED FEATURES****#
# Imbalanced ile başa çıkmanın bir diğer yolu oversampling. Dene training için.
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
conflicting_data["ORIGIN"] = np.where((conflicting_data["ORIGIN"] == 1), 1, 0)
conflicting_data.drop(['ALT', 'REF'], axis=1, inplace=True)

# if conflicting_data["ORIGIN"] == 1, assign 1, else assign 0.


'''
'Feature_type' ,'BIOTYPE' almost all values are same therefore we'll drop these columns
'''
# print("'Feature_type' ,'BIOTYPE' almost all values are same therefore we'll drop these columns")

# print(conflicting_data['BIOTYPE'].value_counts()) #65188 protein_coding, 14 misc_RNA
# print(conflicting_data['Feature_type'].value_counts()) #65172 Transcript, 2 MotiFeature
# print("**************************")
conflicting_data.drop(['BIOTYPE', 'Feature_type'], axis=1, inplace=True)

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

# We'll make CLNDN feature as binary. Whether Tag-value pairs have disease or not. If it has disease yes, else no.
conflicting_data["pairs_has_disease"] = np.where((conflicting_data["CLNDN"] == 'not_specified'), 'no', 'yes')
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

# ENCODING


# categorical_features = ['CHROM', 'IMPACT', 'STRAND', 'BAM_EDIT', 'SIFT', 'PolyPhen',
#                         'BLOSUM62','Consequence']


# We can get the count of unique values for each column.
for col in conflicting_data.columns:
    print(col + ' ' + str(len(conflicting_data[col].unique())) + ' ', (conflicting_data[col]).dtype)
    if (col == "PolyPhen" or col == "IMPACT" or col == "SIFT" or col == "STRAND"):
        print(conflicting_data[col].unique())
    print()

# There is natural order in IMPACT -> ['MODERATE' 'MODIFIER' 'LOW' 'HIGH']
# BAM_EDIT 3 [nan 'OK' 'FAILED'] %51 null %49 OK %1 FAILED
# PolyPhen 5 ['benign' 'probably_damaging' nan 'possibly_damaging' 'unknown']
# SIFT 5 ['tolerated' 'deleterious_low_confidence' 'deleterious' nan 'tolerated_low_confidence']


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# binary_cols = ("ALT","REF")
# for i in binary_cols:
#     column = conflicting_data.iloc[:, conflicting_data.columns.get_loc(i):conflicting_data.columns.get_loc(i) + 1].values
#     conflicting_data.iloc[:, conflicting_data.columns.get_loc(i):conflicting_data.columns.get_loc(i) + 1] = le.fit_transform(column[:,0])


print("--- %s seconds ---" % (time.time() - start_time))
