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



class EEGPowerBandFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, fs=128, nperseg=128):
        self.fs = fs  # Fréquence d'échantillonnage
        self.nperseg = nperseg  # Nombre de points par segment pour le calcul de Welch
        
        # Définition des bandes de fréquences d'intérêt
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    def fit(self, X, y=None):
        return self  # Pas besoin de fitting
    
    def transform(self, X):
        # X est de forme (échantillons, électrodes, temps)
        features = []
        for sample in X:  # Pour chaque époque
            epoch_features = []
            for electrode in sample:  # Pour chaque électrode dans l'époque
                psd, freqs = welch(electrode, fs=self.fs, nperseg=self.nperseg)
                band_powers = []
                for band, (low_freq, high_freq) in self.freq_bands.items():
                    idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                    band_power = np.mean(psd[idx_band])
                    band_powers.append(band_power)
                epoch_features.extend(band_powers)
            features.append(epoch_features)
        return np.array(features)

## Band power
# Band Power + SVM
bp = EEGPowerBandFeatures()
clf = SVC()
pipeline_BP1 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + LR
bp = EEGPowerBandFeatures()
clf = LogisticRegression()
pipeline_BP2 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + Random Forest
bp = EEGPowerBandFeatures()
clf = RandomForestClassifier()
pipeline_BP3 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + Adaboost
bp = EEGPowerBandFeatures()
clf = AdaBoostClassifier()
pipeline_BP4 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + LDA
bp = EEGPowerBandFeatures()
clf = LinearDiscriminantAnalysis()
pipeline_BP5 = Pipeline([('bp', bp), ('clf', clf)])

## XdawnCovariances
# Band Power + XdawnCovariances + TangentSpace + StandardScaler + SVM
bp = EEGPowerBandFeatures()
cov = XdawnCovariances(estimator='lwf')
ts = TangentSpace()
ss = StandardScaler()
clf = SVC()
pipeline_O1 = Pipeline([('bp', bp), ('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Band Power + XdawnCovariances + TangentSpace + StandardScaler + LR
bp = EEGPowerBandFeatures()
cov = XdawnCovariances(estimator='lwf')
ts = TangentSpace()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_O2 = Pipeline([('bp', bp), ('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Band Power + XdawnCovariances + TangentSpace + StandardScaler + Random Forest
bp = EEGPowerBandFeatures()
cov = XdawnCovariances(estimator='lwf')
ts = TangentSpace()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_O3 = Pipeline([('bp', bp), ('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Band Power + XdawnCovariances + TangentSpace + StandardScaler + Adaboost
bp = EEGPowerBandFeatures()
cov = XdawnCovariances(estimator='lwf')
ts = TangentSpace()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_O4 = Pipeline([('bp', bp), ('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Band Power + XdawnCovariances + TangentSpace + StandardScaler + LDA
bp = EEGPowerBandFeatures()
cov = XdawnCovariances(estimator='lwf')
ts = TangentSpace()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_O5 = Pipeline([('bp', bp), ('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

## StandardScaler + Band Power
# StandardScaler + Band Power + SVM
ss = StandardScaler()
bp = EEGPowerBandFeatures()
clf = SVC()
pipeline_SB1 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# StandardScaler + Band Power + LR
ss = StandardScaler()
bp = EEGPowerBandFeatures()
clf = LogisticRegression()
pipeline_SB2 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# StandardScaler + Band Power + Random Forest
ss = StandardScaler()
bp = EEGPowerBandFeatures()
clf = RandomForestClassifier()
pipeline_SB3 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# StandardScaler + Band Power + Adaboost
ss = StandardScaler()
bp = EEGPowerBandFeatures()
clf = AdaBoostClassifier()
pipeline_SB4 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# StandardScaler + Band Power + LDA
ss = StandardScaler()
bp = EEGPowerBandFeatures()
clf = LinearDiscriminantAnalysis()
pipeline_SB5 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

## Wavelet
# Wavelet + SVM
wavelet = EEGWaveletFeatures()
clf = SVC()
pipeline_W1 = Pipeline([('wavelet', wavelet), ('clf', clf)])

# Wavelet + LR
wavelet = EEGWaveletFeatures()
clf = LogisticRegression()
pipeline_W2 = Pipeline([('wavelet', wavelet), ('clf', clf)])

# Wavelet + Random Forest
wavelet = EEGWaveletFeatures()
clf = RandomForestClassifier()
pipeline_W3 = Pipeline([('wavelet', wavelet), ('clf', clf)])

# Wavelet + Adaboost
wavelet = EEGWaveletFeatures()
clf = AdaBoostClassifier()
pipeline_W4 = Pipeline([('wavelet', wavelet), ('clf', clf)])

# Wavelet + LDA
wavelet = EEGWaveletFeatures()
clf = LinearDiscriminantAnalysis()
pipeline_W5 = Pipeline([('wavelet', wavelet), ('clf', clf)])

## Wavelet + SS
# Wavelet + SVM
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
clf = SVC()
pipeline_WS1 = Pipeline([('wavelet', wavelet), ('ss', ss), ('clf', clf)])

# Wavelet + LR
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_WS2 = Pipeline([('wavelet', wavelet), ('ss', ss), ('clf', clf)])

# Wavelet + Random Forest
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_WS3 = Pipeline([('wavelet', wavelet), ('ss', ss), ('clf', clf)])

# Wavelet + Adaboost
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_WS4 = Pipeline([('wavelet', wavelet), ('ss', ss), ('clf', clf)])

# Wavelet + LDA
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_WS5 = Pipeline([('wavelet', wavelet), ('ss', ss), ('clf', clf)])

## Neural Network
# Neural Network
fd = FlattenEEGData()
ss = StandardScaler()
clf = MLPClassifier()
pipeline_NN = Pipeline([('fd', fd), ('ss', ss), ('clf', clf)])



# pipelines

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {'pipeline': [pipeline_BP1],},
    'pipeline_2': {'pipeline': [pipeline_BP2],},
    'pipeline_3': {'pipeline': [pipeline_BP3],},
    'pipeline_4': {'pipeline': [pipeline_BP4],},
    'pipeline_5': {'pipeline': [pipeline_BP5],},
    'pipeline_6': {'pipeline': [pipeline_O1],},
    'pipeline_7': {'pipeline': [pipeline_O2],},
    'pipeline_8': {'pipeline': [pipeline_O3],},
    'pipeline_9': {'pipeline': [pipeline_O4],},
    'pipeline_10': {'pipeline': [pipeline_O5],},
    'pipeline_11': {'pipeline': [pipeline_SB1],},
    'pipeline_12': {'pipeline': [pipeline_SB2],},
    'pipeline_13': {'pipeline': [pipeline_SB3],},
    'pipeline_14': {'pipeline': [pipeline_SB4],},
    'pipeline_15': {'pipeline': [pipeline_SB5],},
    'pipeline_16': {'pipeline': [pipeline_W1],},
    'pipeline_17': {'pipeline': [pipeline_W2],},
    'pipeline_18': {'pipeline': [pipeline_W3],},
    'pipeline_19': {'pipeline': [pipeline_W4],},
    'pipeline_20': {'pipeline': [pipeline_W5],},
    'pipeline_21': {'pipeline': [pipeline_NN],
                    'clf__hidden_layer_sizes': [(64,), (64, 32,), (128, 64, 32,), (256, 128, 64, 32,), (64, 8), (128, 16), (256, 32),],},
    'pipeline_22': {'pipeline': [pipeline_WS1],},
    'pipeline_23': {'pipeline': [pipeline_WS2],},
    'pipeline_24': {'pipeline': [pipeline_WS3],},
    'pipeline_25': {'pipeline': [pipeline_WS4],},
    'pipeline_26': {'pipeline': [pipeline_WS5],},
} # 25*1 + 1*7 = 32 combinations -> 

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
