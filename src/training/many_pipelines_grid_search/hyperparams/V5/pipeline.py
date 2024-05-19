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
import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class FlattenEEGData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Pas besoin de fitting

    def transform(self, X):
        # Aplatir les données de forme [échantillons, électrodes, temps] en [échantillons, électrodes*temps]
        return X.reshape(X.shape[0], -1)


class EEGWaveletFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', levels=5):
        self.wavelet = wavelet  # Type d'ondelette
        self.levels = levels  # Nombre de niveaux de décomposition
        
    def fit(self, X, y=None):
        return self  # Pas besoin de fitting
    
    def transform(self, X):
        # X est de forme (échantillons, électrodes, temps)
        features = []
        for sample in X:  # Pour chaque époque
            epoch_features = []
            for electrode_data in sample:  # Pour chaque électrode dans l'époque
                # Application de la transformation en ondelettes à la série temporelle de l'électrode
                coeffs = pywt.wavedec(electrode_data, self.wavelet, level=self.levels)
                # Extraction et agrégation des coefficients d'ondelette
                electrode_features = []
                for coeff in coeffs:
                    # Calculer des statistiques (par exemple, moyenne et écart-type) pour chaque niveau de décomposition
                    electrode_features.extend([np.mean(coeff), np.std(coeff)])
                epoch_features.extend(electrode_features)
            features.append(epoch_features)
        return np.array(features)


# Wavelet + Random Forest
wavelet = EEGWaveletFeatures()
clf = RandomForestClassifier()
pipeline_W3 = Pipeline([('wavelet', wavelet), ('clf', clf)])


# pipelines

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline_W3],
    }
} # 25*1 + 1*7 = 32 combinations -> 

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
