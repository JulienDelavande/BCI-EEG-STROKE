# This file contain the hyperparameters for the grid search of the pipeline
# Keep the same keys and names for the dictionaries and lists

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class FlattenEEGData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Pas besoin de fitting

    def transform(self, X):
        # Aplatir les données de forme [échantillons, électrodes, temps] en [échantillons, électrodes*temps]
        return X.reshape(X.shape[0], -1)

cov = Covariances()
ts = TangentSpace()
clf = MLPClassifier()
pipeline = Pipeline([('cov', cov), ('ts', ts),  ('clf', clf)])

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline],
    'clf__hidden_layer_sizes': [(64, 32, 16, 8, 4)],
    'clf__activation': ['relu'],
    'clf__solver': ['adam'],
    'clf__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], # 5
    'clf__learning_rate': ['constant', 'invscaling', 'adaptive'], # 3
    'clf__learning_rate_init': [0.0001, 0.001, 0.01, 0.1], # 4
    'clf__max_iter': [100, 200, 300, 400, 500] # 5
    }, # 1*1*1*5*3*4*5 = 300
    #'pipeline_2':{}
} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
