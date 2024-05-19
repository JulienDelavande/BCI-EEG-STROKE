# This file contain the hyperparameters for the grid search of the pipeline
# Keep the same keys and names for the dictionaries and lists

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = SVC()
pipeline = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline],
    'cov__estimator': ['oas'], # 1
    },
    #'pipeline_2':{}
} # 1*1 = 1 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
