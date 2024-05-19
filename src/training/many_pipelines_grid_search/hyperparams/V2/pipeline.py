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

# Covariance + Tangent Space + Standard Scaler + LR
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_R4 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

## CSP
# CSP + LR
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = LogisticRegression()
pipeline_CSP4 = Pipeline([('csp', csp), ('clf', clf)])

## xDAWN
# xDAWN + Vectorizer + Scaler + LR
xdawn = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_X4 = Pipeline([('xdawn', xdawn), ('vec', vec), ('ss', ss), ('clf', clf)])


## other
# Vectorizer + Scaler + LR
vec = MyVectorizer()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_O4 = Pipeline([('vec', vec), ('ss', ss), ('clf', clf)])

# pipelines

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline_R4],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_2': {
    'pipeline': [pipeline_CSP4],
    },
    'pipeline_3': {
    'pipeline': [pipeline_X4],
    },
    'pipeline_4': {
    'pipeline': [pipeline_O4],
    }
} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
