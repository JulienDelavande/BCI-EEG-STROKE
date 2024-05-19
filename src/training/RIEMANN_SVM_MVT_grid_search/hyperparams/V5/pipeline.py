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

pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline],
    'clf__C': [0.01, 0.1, 0.2, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 1000], # 17
    'clf__gamma': [0.01, 0.1, 0.2, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 1000, 'scale'], # 18
    'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid'], # 4
    'cov__estimator': ['oas', 'lwf', 'scm'], # 3
    'ts__metric': ['riemann', 'logeuclid', 'euclid'], # 3
    'ss__with_mean': [True, False], # 2
    },
    #'pipeline_2':{}
} # 17*18*4*3*3*2 = 22032 combinations -> 8 nodes, 6cores/node, 7344/48 = 459

pipelines_dict_lists_exclude = None
pipelines_exclude_rules = None
