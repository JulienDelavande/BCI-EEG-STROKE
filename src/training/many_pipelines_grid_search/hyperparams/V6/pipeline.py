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

class MyVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.features_shape_ = X.shape[1:]
        return self

    def transform(self, X, y=None):
        return X.reshape(len(X), -1)


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


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import mne

class BandPowerExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=1024, freq_bands=[(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)], n_samples=3073):
        self.sfreq = sfreq
        self.freq_bands = freq_bands
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = np.empty((X.shape[0], len(self.freq_bands)))
        for i, epoch in enumerate(X):
            # Redimensionner l'époque en tableau 3D
            epoch = np.reshape(epoch, (1, 37, -1))
            if i == 0:
                print(f'epoch.shape: {epoch.shape}')
            power, _ = mne.time_frequency.psd_array_multitaper(epoch, fmin=self.freq_bands[0][0], fmax=self.freq_bands[-1][1], sfreq=1024)
            if i == 0:
                print(f'power.shape: {power.shape}')
            for j, freq_band in enumerate(self.freq_bands):
                X_transformed[i, j] = np.mean(power[:, freq_band[0]:freq_band[1]], axis=(1, 2))
            if i == 0:
                print(f'X_transformed[i]: {X_transformed[i]}')
        return X_transformed

## Tangent Space ########################################################
# Tangent Space + SVM
cov = Covariances()
ts = TangentSpace()
clf = SVC()
pipeline_CT1 = Pipeline([('cov', cov), ('ts', ts), ('clf', clf)])

# Tangent Space + LR
cov = Covariances()
ts = TangentSpace()
clf = LogisticRegression()
pipeline_CT2 = Pipeline([('cov', cov), ('ts', ts), ('clf', clf)])

# Tangent Space + Random Forest
cov = Covariances()
ts = TangentSpace()
clf = RandomForestClassifier()
pipeline_CT3 = Pipeline([('cov', cov), ('ts', ts), ('clf', clf)])

# Tangent Space + Adaboost
cov = Covariances()
ts = TangentSpace()
clf = AdaBoostClassifier()
pipeline_CT4 = Pipeline([('cov', cov), ('ts', ts), ('clf', clf)])

# Tangent Space + LDA
cov = Covariances()
ts = TangentSpace()
clf = LinearDiscriminantAnalysis()
pipeline_CT5 = Pipeline([('cov', cov), ('ts', ts), ('clf', clf)])

# Tangent Space + MLP
cov = Covariances()
ts = TangentSpace()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_CT6 = Pipeline([('cov', cov), ('ts', ts), ('vec', vec), ('clf', clf)])


## Cov + ts + SS ########################################################
# Cov + ts + ss + SVM
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = SVC()
pipeline_CTS1 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Cov + ts + ss + LR
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler() 
clf = LogisticRegression()
pipeline_CTS2 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Cov + ts + ss + Random Forest
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_CTS3 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Cov + ts + ss + Adaboost
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_CTS4 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Cov + ts + ss + LDA
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_CTS5 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('clf', clf)])

# Cov + ts + ss + MLP
cov = Covariances()
ts = TangentSpace()
ss = StandardScaler()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_CTS6 = Pipeline([('cov', cov), ('ts', ts), ('ss', ss), ('vec', vec), ('clf', clf)])


## Band Power ########################################################
# Band Power + ss + SVM
bp = BandPowerExtractor()
ss = StandardScaler()
clf = SVC()
pipeline_BPS1 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# Band Power + ss + LR
bp = BandPowerExtractor()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_BPS2 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# Band Power + ss + Random Forest
bp = BandPowerExtractor()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_BPS3 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# Band Power + ss + Adaboost
bp = BandPowerExtractor()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_BPS4 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# Band Power + ss + LDA
bp = BandPowerExtractor()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_BPS5 = Pipeline([('bp', bp), ('ss', ss), ('clf', clf)])

# Band Power + ss + MLP
bp = BandPowerExtractor()
ss = StandardScaler()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_BPS6 = Pipeline([('bp', bp), ('ss', ss), ('vec', vec), ('clf', clf)])


## Band power ########################################################
# Band Power + SVM
bp = BandPowerExtractor()
clf = SVC()
pipeline_BP1 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + LR
bp = BandPowerExtractor()
clf = LogisticRegression()
pipeline_BP2 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + Random Forest
bp = BandPowerExtractor()
clf = RandomForestClassifier()
pipeline_BP3 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + Adaboost
bp = BandPowerExtractor()
clf = AdaBoostClassifier()
pipeline_BP4 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + LDA
bp = BandPowerExtractor()
clf = LinearDiscriminantAnalysis()
pipeline_BP5 = Pipeline([('bp', bp), ('clf', clf)])

# Band Power + MLP
bp = BandPowerExtractor()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_BP6 = Pipeline([('bp', bp), ('vec', vec), ('clf', clf)])


## Band Power + Xdawn ########################################################
bp = BandPowerExtractor()
cov = XdawnCovariances()
clf = SVC()
pipeline_BPX1 = Pipeline([('bp', bp), ('cov', cov), ('clf', clf)])

bp = BandPowerExtractor()
cov = XdawnCovariances()
clf = LogisticRegression()
pipeline_BPX2 = Pipeline([('bp', bp), ('cov', cov), ('clf', clf)])

bp = BandPowerExtractor()
cov = XdawnCovariances()
clf = RandomForestClassifier()
pipeline_BPX3 = Pipeline([('bp', bp), ('cov', cov), ('clf', clf)])

bp = BandPowerExtractor()
cov = XdawnCovariances()
clf = AdaBoostClassifier()
pipeline_BPX4 = Pipeline([('bp', bp), ('cov', cov), ('clf', clf)])

bp = BandPowerExtractor()
cov = XdawnCovariances()
clf = LinearDiscriminantAnalysis()
pipeline_BPX5 = Pipeline([('bp', bp), ('cov', cov), ('clf', clf)])

bp = BandPowerExtractor()
cov = XdawnCovariances()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_BPX6 = Pipeline([('bp', bp), ('cov', cov), ('vec', vec), ('clf', clf)])


## Wavelet ########################################################
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

# Wavelet + MLP
wavelet = EEGWaveletFeatures()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_W6 = Pipeline([('wavelet', wavelet), ('vec', vec), ('clf', clf)])

## Wavelet + SS ########################################################
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

# Wavelet + MLP
wavelet = EEGWaveletFeatures()
ss = StandardScaler()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_WS6 = Pipeline([('wavelet', wavelet), ('ss', ss), ('vec', vec), ('clf', clf)])

## Xdawn ########################################################
# Xdawn + SVM
cov = XdawnCovariances()
clf = SVC()
pipeline_X1 = Pipeline([('cov', cov), ('clf', clf)])

# Xdawn + LR
cov = XdawnCovariances()
clf = LogisticRegression()
pipeline_X2 = Pipeline([('cov', cov), ('clf', clf)])

# Xdawn + Random Forest
cov = XdawnCovariances()
clf = RandomForestClassifier()
pipeline_X3 = Pipeline([('cov', cov), ('clf', clf)])

# Xdawn + Adaboost
cov = XdawnCovariances()
clf = AdaBoostClassifier()
pipeline_X4 = Pipeline([('cov', cov), ('clf', clf)])

# Xdawn + LDA
cov = XdawnCovariances()
clf = LinearDiscriminantAnalysis()
pipeline_X5 = Pipeline([('cov', cov), ('clf', clf)])

# Xdawn + MLP
cov = XdawnCovariances()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_X6 = Pipeline([('cov', cov), ('vec', vec), ('clf', clf)]) 

## Xdawn + vec + SS ########################################################
# Xdawn + ss + SVM
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = SVC()
pipeline_XS1 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

# Xdawn + ss + LR
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_XS2 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

# Xdawn + ss + Random Forest
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_XS3 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

# Xdawn + ss + Adaboost
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_XS4 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

# Xdawn + ss + LDA
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_XS5 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

# Xdawn + ss + MLP
cov = XdawnCovariances()
vec = MyVectorizer()
ss = StandardScaler()
clf = MLPClassifier()
pipeline_XS6 = Pipeline([('cov', cov), ('vec', vec), ('ss', ss), ('clf', clf)])

## CSP ########################################################
# CSP + SVM
csp = CSP()
clf = SVC()
pipeline_C1 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + LR
csp = CSP()
clf = LogisticRegression()
pipeline_C2 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + Random Forest
csp = CSP()
clf = RandomForestClassifier()
pipeline_C3 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + Adaboost
csp = CSP()
clf = AdaBoostClassifier()
pipeline_C4 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + LDA
csp = CSP()
clf = LinearDiscriminantAnalysis()
pipeline_C5 = Pipeline([('csp', csp), ('clf', clf)])

# CSP + MLP
csp = CSP()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_C6 = Pipeline([('csp', csp), ('vec', vec), ('clf', clf)])

## CSP + SS ########################################################
# CSP + ss + SVM
csp = CSP()
ss = StandardScaler()
clf = SVC()
pipeline_CS1 = Pipeline([('csp', csp), ('ss', ss), ('clf', clf)])

# CSP + ss + LR
csp = CSP()
ss = StandardScaler()
clf = LogisticRegression()
pipeline_CS2 = Pipeline([('csp', csp), ('ss', ss), ('clf', clf)])

# CSP + ss + Random Forest
csp = CSP()
ss = StandardScaler()
clf = RandomForestClassifier()
pipeline_CS3 = Pipeline([('csp', csp), ('ss', ss), ('clf', clf)])

# CSP + ss + Adaboost
csp = CSP()
ss = StandardScaler()
clf = AdaBoostClassifier()
pipeline_CS4 = Pipeline([('csp', csp), ('ss', ss), ('clf', clf)])

# CSP + ss + LDA
csp = CSP()
ss = StandardScaler()
clf = LinearDiscriminantAnalysis()
pipeline_CS5 = Pipeline([('csp', csp), ('ss', ss), ('clf', clf)])

# CSP + ss + MLP
csp = CSP()
ss = StandardScaler()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_CS6 = Pipeline([('csp', csp), ('ss', ss), ('vec', vec), ('clf', clf)])

## CSP + Xdawn ########################################################
# CSP + Xdawn + SVM
csp = CSP()
cov = XdawnCovariances()
clf = SVC()
pipeline_CX1 = Pipeline([('csp', csp), ('cov', cov), ('clf', clf)])

# CSP + Xdawn + LR
csp = CSP()
cov = XdawnCovariances()
clf = LogisticRegression()
pipeline_CX2 = Pipeline([('csp', csp), ('cov', cov), ('clf', clf)])

# CSP + Xdawn + Random Forest
csp = CSP()
cov = XdawnCovariances()
clf = RandomForestClassifier()
pipeline_CX3 = Pipeline([('csp', csp), ('cov', cov), ('clf', clf)])

# CSP + Xdawn + Adaboost
csp = CSP()
cov = XdawnCovariances()
clf = AdaBoostClassifier()
pipeline_CX4 = Pipeline([('csp', csp), ('cov', cov), ('clf', clf)])

# CSP + Xdawn + LDA
csp = CSP()
cov = XdawnCovariances()
clf = LinearDiscriminantAnalysis()
pipeline_CX5 = Pipeline([('csp', csp), ('cov', cov), ('clf', clf)])

# CSP + Xdawn + MLP
csp = CSP()
cov = XdawnCovariances()
vec  = FlattenEEGData()
clf = MLPClassifier()
pipeline_CX6 = Pipeline([('csp', csp), ('cov', cov), ('vec', vec), ('clf', clf)])

# pipelines

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {'pipeline': [pipeline_CT1]},
    'pipeline_2': {'pipeline': [pipeline_CT2]},
    'pipeline_3': {'pipeline': [pipeline_CT3]},
    'pipeline_4': {'pipeline': [pipeline_CT4]},
    'pipeline_5': {'pipeline': [pipeline_CT5]},
    'pipeline_6': {'pipeline': [pipeline_CT6]},
    'pipeline_7': {'pipeline': [pipeline_CTS1]},
    'pipeline_8': {'pipeline': [pipeline_CTS2]},
    'pipeline_9': {'pipeline': [pipeline_CTS3]},
    'pipeline_10': {'pipeline': [pipeline_CTS4]},
    'pipeline_11': {'pipeline': [pipeline_CTS5]},
    'pipeline_12': {'pipeline': [pipeline_CTS6]},
    'pipeline_13': {'pipeline': [pipeline_BPS1]},
    'pipeline_14': {'pipeline': [pipeline_BPS2]},
    'pipeline_15': {'pipeline': [pipeline_BPS3]},
    'pipeline_16': {'pipeline': [pipeline_BPS4]},
    'pipeline_17': {'pipeline': [pipeline_BPS5]},
    'pipeline_18': {'pipeline': [pipeline_BPS6]},
    'pipeline_19': {'pipeline': [pipeline_BP1]},
    'pipeline_20': {'pipeline': [pipeline_BP2]},
    'pipeline_21': {'pipeline': [pipeline_BP3]},
    'pipeline_22': {'pipeline': [pipeline_BP4]},
    'pipeline_23': {'pipeline': [pipeline_BP5]},
    'pipeline_24': {'pipeline': [pipeline_BP6]},
    'pipeline_25': {'pipeline': [pipeline_BPX1]},
    'pipeline_26': {'pipeline': [pipeline_BPX2]},
    'pipeline_27': {'pipeline': [pipeline_BPX3]},
    'pipeline_28': {'pipeline': [pipeline_BPX4]},
    'pipeline_29': {'pipeline': [pipeline_BPX5]},
    'pipeline_30': {'pipeline': [pipeline_BPX6]},
    'pipeline_31': {'pipeline': [pipeline_W1]},
    'pipeline_32': {'pipeline': [pipeline_W2]},
    'pipeline_33': {'pipeline': [pipeline_W3]},
    'pipeline_34': {'pipeline': [pipeline_W4]},
    'pipeline_35': {'pipeline': [pipeline_W5]},
    'pipeline_36': {'pipeline': [pipeline_W6]},
    'pipeline_37': {'pipeline': [pipeline_WS1]},
    'pipeline_38': {'pipeline': [pipeline_WS2]},
    'pipeline_39': {'pipeline': [pipeline_WS3]},
    'pipeline_40': {'pipeline': [pipeline_WS4]},
    'pipeline_41': {'pipeline': [pipeline_WS5]},
    'pipeline_42': {'pipeline': [pipeline_WS6]},
    'pipeline_43': {'pipeline': [pipeline_X1]},
    'pipeline_44': {'pipeline': [pipeline_X2]},
    'pipeline_45': {'pipeline': [pipeline_X3]},
    'pipeline_46': {'pipeline': [pipeline_X4]},
    'pipeline_47': {'pipeline': [pipeline_X5]},
    'pipeline_48': {'pipeline': [pipeline_X6]},
    'pipeline_49': {'pipeline': [pipeline_XS1]},
    'pipeline_50': {'pipeline': [pipeline_XS2]},
    'pipeline_51': {'pipeline': [pipeline_XS3]},
    'pipeline_52': {'pipeline': [pipeline_XS4]},
    'pipeline_53': {'pipeline': [pipeline_XS5]},
    'pipeline_54': {'pipeline': [pipeline_XS6]},
    'pipeline_55': {'pipeline': [pipeline_C1]},
    'pipeline_56': {'pipeline': [pipeline_C2]},
    'pipeline_57': {'pipeline': [pipeline_C3]},
    'pipeline_58': {'pipeline': [pipeline_C4]},
    'pipeline_59': {'pipeline': [pipeline_C5]},
    'pipeline_60': {'pipeline': [pipeline_C6]},
    'pipeline_61': {'pipeline': [pipeline_CS1]},
    'pipeline_62': {'pipeline': [pipeline_CS2]},
    'pipeline_63': {'pipeline': [pipeline_CS3]},
    'pipeline_64': {'pipeline': [pipeline_CS4]},
    'pipeline_65': {'pipeline': [pipeline_CS5]},
    'pipeline_66': {'pipeline': [pipeline_CS6]},
    'pipeline_67': {'pipeline': [pipeline_CX1]},
    'pipeline_68': {'pipeline': [pipeline_CX2]},
    'pipeline_69': {'pipeline': [pipeline_CX3]},
    'pipeline_70': {'pipeline': [pipeline_CX4]},
    'pipeline_71': {'pipeline': [pipeline_CX5]},
    'pipeline_72': {'pipeline': [pipeline_CX6]},
}

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
