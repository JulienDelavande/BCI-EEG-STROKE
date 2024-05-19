# This file contain the hyperparameters for the grid search of the pipeline
# Keep the same keys and names for the dictionaries and lists

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import XdawnCovariances
from sklearn.linear_model import LogisticRegression

class MyVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.features_shape_ = X.shape[1:]
        return self

    def transform(self, X, y=None):
        return X.reshape(len(X), -1)

## Riemannian Geometry
# Covariance + Tangent Space + Standard Scaler + Random Forest
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_R1 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Covariance + Tangent Space + Standard Scaler + SVM
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = SVC()
pipeline_R2 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Covariance + Tangent Space + Standard Scaler + LDA
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_R3 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Covariance + Tangent Space + Standard Scaler + LR
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_R4 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Covariance + Tangent Space + Standard Scaler + Adaboost
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_R5 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])


## CSP
# CSP + Random Forest
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = RandomForestClassifier()
pipeline_CSP1 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + SVM
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = SVC()
pipeline_CSP2 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + LDA
lda = LinearDiscriminantAnalysis()                            
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
pipeline_CSP3 = Pipeline([('csp', csp), ('clf', lda)])

# CSP + LR
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = LogisticRegression()
pipeline_CSP4 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + Adaboost
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = AdaBoostClassifier()
pipeline_CSP5 = Pipeline([('csp', csp), ('clf', clf)])


## xDAWN
# xDAWN + Vectorizer + Scaler + Random Forest
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_X1 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])

# xDAWN + Vectorizer + Scaler + SVM
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = SVC()
pipeline_X2 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])

# xDAWN + Vectorizer + Scaler + LDA
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_X3 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])

# xDAWN + Vectorizer + Scaler + LR
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_X4 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])

# xDAWN + Vectorizer + Scaler + Adaboost
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_X5 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])


## other
# Vectorizer + Scaler + Random Forest
vec = MyVectorizer()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_O1 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# Vectorizer + Scaler + SVM
vec = MyVectorizer()
ss = StandardScaler()
clf = SVC()
pipeline_O2 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# Vectorizer + Scaler + LDA
vec = MyVectorizer()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_O3 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# Vectorizer + Scaler + LR
vec = MyVectorizer()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_O4 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# Vectorizer + Scaler + Adaboost
vec = MyVectorizer()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_O5 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# pipelines

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline_R1],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_2': {
    'pipeline': [pipeline_R2],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_3': {
    'pipeline': [pipeline_R3],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_4': {
    'pipeline': [pipeline_R4],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_5': {
    'pipeline': [pipeline_R5],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_6': {
    'pipeline': [pipeline_CSP1],
    },
    'pipeline_7': {
    'pipeline': [pipeline_CSP2],
    },
    'pipeline_8': {
    'pipeline': [pipeline_CSP3],
    },
    'pipeline_9': {
    'pipeline': [pipeline_CSP4],
    },
    'pipeline_10': {
    'pipeline': [pipeline_CSP5],
    },
    'pipeline_11': {
    'pipeline': [pipeline_X1],
    },
    'pipeline_12': {
    'pipeline': [pipeline_X2],
    },
    'pipeline_13': {
    'pipeline': [pipeline_X3],
    },
    'pipeline_14': {
    'pipeline': [pipeline_X4],
    },
    'pipeline_15': {
    'pipeline': [pipeline_X5],
    },
    'pipeline_16': {
    'pipeline': [pipeline_O1],
    },
    'pipeline_17': {
    'pipeline': [pipeline_O2],
    },
    'pipeline_18': {
    'pipeline': [pipeline_O3],
    },
    'pipeline_19': {
    'pipeline': [pipeline_O4],
    },
    'pipeline_20': {
    'pipeline': [pipeline_O5],
    },

} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
