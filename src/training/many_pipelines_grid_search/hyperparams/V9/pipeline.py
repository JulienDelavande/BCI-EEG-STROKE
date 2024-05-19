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
import mne

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    def __init__(self, num_electrodes=37, num_time_points=3073, num_classes=2):
        super(TemporalCNN, self).__init__()
        # Les dimensions d'entrée sont (batch_size, 1, 37, 3073)
        # Première couche de convolution qui applique 16 filtres
        self.conv1 = nn.Conv2d(1, 16, (num_electrodes, 5), padding='same')
        # Première couche de pooling réduisant la dimension temporelle
        self.pool = nn.MaxPool2d((1, 4))  # Réduction de la dimension temporelle par 4
        # Seconde couche de convolution qui applique 32 filtres
        self.conv2 = nn.Conv2d(16, 32, (1, 5), padding='same')
        # Seconde couche de pooling réduisant davantage la dimension temporelle
        self.pool2 = nn.MaxPool2d((1, 4))
        
        # Ajustez la taille ici en fonction de la réduction effectuée par les couches de pooling
        # Par exemple, si après les deux poolings la taille est réduite à 1/16 de 3073
        # la nouvelle taille serait 3073 / 4 / 4 = 192
        taille_temporelle_après_pooling = 37 * 192
        
        # Couche dense
        self.fc1 = nn.Linear(32 * taille_temporelle_après_pooling, 128)
        # Couche de sortie
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, 37, 3073)
        # Appliquer la première convolution + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Appliquer la seconde convolution + pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # Aplatir les caractéristiques pour les couches denses
        x = x.view(x.size(0), -1)
        # Couches denses
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes=2, epochs=3, batch_size=32, learning_rate=1e-3):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = TemporalCNN(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        loss_epoch = []
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            loss_epoch.append(running_loss / len(loader))
            print(f'Epoch {epoch + 1}/{self.epochs} - Loss: {running_loss / len(loader)}')
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(X)
        return output.argmax(dim=1).numpy()


# pipelines
pipeline = Pipeline([('clf', PyTorchCNNClassifier())])

# pipelines_dict_lists : paramteres to be used in the grid search for the pipeline
pipelines_dict_lists = {
    'pipeline_1': {
    'pipeline': [pipeline],
    'clf__num_classes': [2],
    'clf__epochs': [3],
    'clf__batch_size': [32],
    'clf__learning_rate': [1e-3, 1e-4],
    },
} # 2*2*1*1*1 = 4 combinations -> 8 nodes, 6cores/node

# pipelines_dict_lists_exclude : parameters to be excluded from the grid search for the pipeline (same keys as pipelines_dict_lists)
pipelines_dict_lists_exclude = None

# pipelines_exclude_rules : rules to exclude some pipelines combinations
# Example: pipelines_exclude_rules = [lambda params: params['clf__C'] >= params['clf__gamma'], ...]
pipelines_exclude_rules = None
