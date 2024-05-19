from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
svm = SVC()
lr = LogisticRegression()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()

pipeline_svm = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', svm)])
pipeline_lr = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', lr)])
pipeline_rf = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', rf)])
pipeline_ab = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', ab)])

pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline_svm],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_2':{
    'pipeline': [pipeline_lr],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_3':{
    'pipeline': [pipeline_rf],
    'cov__estimator': ['oas'], # 1
    },
    'pipeline_4':{
    'pipeline': [pipeline_ab],
    'cov__estimator': ['oas'], # 1
    }
}

pipelines_dict_lists_exclude = None