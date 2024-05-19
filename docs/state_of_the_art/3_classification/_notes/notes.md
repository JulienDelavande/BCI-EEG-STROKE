# 3 - Classification : notes about articles

## TSC (Time Series Classification) 

### What is said in litterature

From Fawaz & al. (2018) (p.930), CNN (and more generally DNN) seem to be the most used and promising type of classifier for physiological time-series classification such as EEG features signals. 

Fawaz & al. (2018) conducted an analysis to compare 9 DNN architectures: 

![Table1 from /img](img/Table1 "Table1")

with hyperparameters optimization as:

![Table2 from /img](img/Table2 "Table2")

Find the implementation and analysis of the classifiers on the github page associated to the paper: [https://github.com/hfawaz/dl-4-tsc]()

### What about us ?

The data we deal with in MTS (multivariate time series). Tested architectures will have to be able to deal with MTS. Moreover, we should prefer architecture with high degree of transferability (transfert learning) try speed up training when adapting to new patient.

We should try and compare the following approaches for EEG MTS signals:

* ResNet
* FCN
* (Encoder)
