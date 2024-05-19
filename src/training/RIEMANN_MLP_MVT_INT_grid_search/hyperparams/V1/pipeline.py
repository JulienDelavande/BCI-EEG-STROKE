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
    'clf__hidden_layer_sizes': [(100, 100), 
                                (100, 100, 100), 
                                (100, 100, 100, 100), 
                                (100, 100, 100, 100, 100),
                                (8, 4),
                                (8, 4, 2),
                                (16, 8),
                                (16, 8, 4),
                                (16, 8, 4, 2),
                                (32, 16),
                                (32, 16, 8),
                                (32, 16, 8, 4),
                                (32, 16, 8, 4, 2),
                                (64, 32),
                                (64, 32, 16),
                                (64, 32, 16, 8),
                                (64, 32, 16, 8, 4),
                                (128, 64),
                                (128, 64, 32),
                                (128, 64, 32, 16),
                                (128, 64, 32, 16, 8)
                                ],
    'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
    },
    #'pipeline_2':{}
} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
