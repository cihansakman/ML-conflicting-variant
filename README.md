<h1 align="center"> Genetic Variant Classification </h1>

<h4 align="center">Abstract</h4>
<p align="center">
In this project, we’ll build Machine Learning algorithms to predict whether a given variant has <b>conflicting classification</b> or not based on the dataset obtained from <a href="https://www.kaggle.com/kevinarvai/clinvar-conflicting">kaggle</a>. During this project we’ll commonly use <b>scikit-learn</b> library in Python. While applying Machine Learning Algorithms we’ll build four different algorithms such as: Gradient Boost, Logistic Regression, Decision Tree, and Random Forest classifiers.
</p>

## Introduction
The main purpose of this project is predicting given human genetic variant have conflicting classification or not based on Machine Learning classification algorithms. Variants that have con- flicting classifications (from laboratory to laboratory) can confuse when clinicians or researchers try to interpret whether the variant impacts the disease of a given patient. If we gain a good classification algorithm at the end of this project, we would like to help clinicians detect the variants classification

Necessary information and the data itself can be achieved from: [kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting).

## Preprocessing Steps

The data has **65188** rows and **48** columns.

### Clean Fully NaN Features

First of all, there were **10** columns which has %99.9 NaN values. These columns have cleaned from the dataset.

### Convert Into Float

In cDNA position, CDS position, Protein position values are integer but type of features are Object. Because there are some approximation such as 1331-1332 for relative positions. We took the valid(first or second) one because they are already so close to each others.

We can see the changes below.

Before preprocessing       |  After preprocessing
:-------------------------:|:-------------------------:
![](images/cdna_before.png)  |  ![](images/cdna_after.png)

###	CHROM Feature

CHROM represented for chromosome the variant has located on, and we have 22 + 2 chromosome in the dataset. The first 22 chromosomes are represented from 1 to 22, and the other two chromosomes are Y and Mitochondrial Chromosome. These two are represented as Y and MT. These two chromosomes have converted into 23 and 24.

###	Highly Correlated Features

First, correlation matrix has been plotted and the correlation between features analyzed. After analyzing correlation matrix we clearly see that ’cDNA position’, ’CDS position’,’Protein position’ features has 1 correlation between them. We can drop two of them and use just cDNA position because it has fewer NaN values than others.

<div align="center"

![Correlation Matrix](images/cormatrix.png "Correlation Matrix")

</div>

###	EXON and INTRON

These two columns represent the exon and intron number out of the total number. These values are kept as i.e. 10/12. These values are converted into float numbers and kept as float.

The other issue for these features were NaN values. For EXON  and  INTRON,  their  NaN ratios are parallel. The percentage of NaN values for INTRON is %86.4, and the percentage of NaN values for EXON is %13.6. Therefore the NaN values of EXON with filled with the corresponding values of INTRON.

### Zero Varience

In the dataset, two columns have zero variance(all variables are unique). These two columns are *CLNHGVS*, and *CLNVI* with the Object type. In the CLNVI feature, there are 27659 non-null variables and 27655 unique values. In the CLNHGVS feature, there are 65188 unique values out of 65188. These two columns dropped from the dataset.

###	Similar Features

**ALT**(Alternate allele) and Allele are referring to the same alleles in the human genetic. Therefore we can drop the Allele column because it has some NaN values, but ALT doesn’t.

**MC** feature is the comma-separated list of molecular Consequence in the form of Sequence Ontology, and Consequence feature is the type of molecular Consequence. Therefore, we will keep the type of molecular Sequence and drop the MC feature.

**CLNDISDB** feature is the tag-value pairs of disease database name and identifier, and **CLNDN** feature is the disease name for the identifier. Therefore we will keep the disease name and drop the **CLNDISDB**. At the same time, in the **CLNDN** column, both *not_provided* and *not_specified* values are referring the same meaning. Both values were kept as *not_specified*. For last, the CLNDN column has been kept as a binary column. Whether tag-value pairs have a disease or not. If it has a disease, assign 1, else 0.

### Dealing With Missing Values

#### Missing Because not Exist
cDNA can be described as DNA without all the necessary noncoding regions. That is means if there is no EXON, we cannot talk about the cDNA position. There is an exception. In the lack of EXON and INTRON, we may have a cDNA position. Therefore we will fill the NaN values in **cDNA_position** as -1. That will refer to there is no such position.

CADD is a tool for scoring the deleteriousness of single nucleotide variants and insertion/deletion variants in the human genome. All the NaN values in **CADD_PHRED** and **CADD_RAW** belong to non-single nucleotide variants. That is means there is no CADD information for non-single nucleotide variants. We will fill the values with -1.

In the dataset, **AminoAcids** and **Codons** are only given if the variant affects protein-coding sequence. Therefore, NaN values in AminoAcids and Codons columns filled with **not-affect**.

#### Nominal Missing Values

There is only one case for the Nominal Missing Values and that is for **LoFtool**. Missing values in LoFtool column filled with the mean by using **sklearn** impute library.

##	Imbalanced Classification

We can clerly see that our target class is imbalanced. Before fit our training dataset we used a sampling algorithm called **Under Sampling**. We can observe the changes in the target class before and after the undersampling.

*Before Sampling*            |  *After Sampling*
:-------------------------:|:-------------------------:
![](images/before_sampling.png)  |  ![](images/after_sampling.png)

## Apply Machine Learning Algorithms

After preprocessing we prepared our dataset for applying learning algorithms. We wanted to build four different learning algorithms to find the best for our dataset. We will use the following algorithms: Gradient Boosting Classifier, Logistic Regression, Random Forest, and Decision Tree Classifiers. Our aim to use these algorithms is that predicting the probability of CLASS(conflicting or not) membership for each example. Before these predictions, we can plot the importance of features.

<div align="center"

![Importance of Features](images/importance.png "Importance of 15 Features")

</div>

### Performance Metrics

Area Under ROC Curve (or ROC AUC for short) is a performance metric for binary classification problems and it is suitable for the predict the probability of the target class. A ROC Curve is a plot of the true positive rate and the false positive rate for a given set of probability predictions at different thresholds used to map the probabilities to class labels. The area under the curve is then the approximate integral **under the ROC Curve(AUC)**. Also we’ll print the **precision**, **recall** and **f1-scores** of the algorithms.

## Comparision of AUC Scores

In below figure we can better see which learning algorithm is better for our dataset.

<div align="center"

![Comparision of AUC Scores](images/all_ml.png "Comparision of AUC Scores")

</div>

According to the above figure, Gradient Boost Classifier is the best option for our algorithm. We have a 0.78 AUC score. It would be better if we had over 0.85 AUC score. This algorithm still needs to be improved to use for real life problem.

## Conclusion 

In conclusion, we tried to deal with Human Genetic Variant and make a prediction about whether a variant have conflicting. Due to this is a real-world problem our dataset was dirty and we had to clean up the data in preprocessing section. We dropped fully NaN, duplicate, and dummy features from our dataset. After that, we tried to fill NaN values with different approaches. At the end of the preprocessing, we applied OneHotEncoding, Mapping and Feature Hashing to transforming categorical values into numeric values.

After preparing our dataset, we applied four different classification algorithms to find the best algorithm. After applying all algorithms, we could not achieve as good results as we wanted. For improving our accuracy scores, we applied the PCA algorithm to reduce dimension and have better accuracy scores. Nevertheless, PCA would not be beneficial for our algorithms. As a result, we conclude that for this learning algorithm Gradient Boosting Classifier had the best accuracy score with the best parameter settings
