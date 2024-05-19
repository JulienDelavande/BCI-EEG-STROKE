# This file contain the hyperparameters for the grid search of the pipeline
# Keep the same keys and names for the dictionaries and lists

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .settings import settings

RANDOM_STATE = settings['RANDOM_STATE']

cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = SVC(random_state=RANDOM_STATE) # Add random state for reproducibility for each element in the pipeline that has a random state
pipeline = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline],
    'clf__C': [1, 10], # 2
    'clf__gamma': [1, 10, 'scale'], # 2
    'clf__kernel': ['rbf'], # 1
    'cov__estimator': ['oas'], # 1
    },
    #'pipeline_2':{}
} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
