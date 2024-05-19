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
    'clf__C': [1], # 1
    'clf__gamma': ['scale'], # 1
    'clf__kernel': ['rbf'], # 1
    'cov__estimator': ['oas'], # 1
    },
    #'pipeline_2':{}
}

pipelines_dict_lists_exclude = None
pipelines_exclude_rules = None
