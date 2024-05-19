"""
Module to define CNN models, starting with a base model.
"""

import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    """
    """

    def __init__(
            self,
            nb_channels: int,
            windows_len: int) -> None:
        super(BaseCNN, self).__init__()

        CHANNEL_SIZE_I = 1
        CHANNEL_SIZE_2 = 16
        CHANNEL_SIZE_3 = 8
        CHANNEL_SIZE_4 = 4

        self.net = nn.Sequential(
            # Input size = ( 1,   nb_channels,   windows_len)
            nn.Conv2d(CHANNEL_SIZE_I,
                      CHANNEL_SIZE_2,
                      kernel_size=(nb_channels, 1),
                      padding="valid"),
            # Input size = ( 1,   1,   windows_len)
            nn.BatchNorm2d(CHANNEL_SIZE_2),
            # Input size = ( 1,   1,   windows_len)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Dropout(0.5),

            # Input size = (16, nb_channels/2, windows_len/2)
            nn.Conv2d(CHANNEL_SIZE_2,
                      CHANNEL_SIZE_3,
                      kernel_size=(1, 32),
                      dilation=(1, 2),
                      padding="same"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Input size = ( 8, nb_channels/2, windows_len/4)
            nn.Conv2d(8,  4, kernel_size=(5, 5), dilation=(2, 2)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.3),
            nn.Linear(256,   2),
            nn.Softmax(dim=1))




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    LeakyReLU,
    Dense,
)


def generate_baseCNN(nbr_channel, windows_len):
    """
    """
    # Taille de la frame: FRAME = windows_size * sampling rate de l'EEG (1024 pour nous)
    model = Sequential()
    model.add(InputLayer(input_shape=(1, nbr_channel, windows_len)))
    # La première couche de convolution est spatiale
    # (1, nbr_channel, windows_len)
    model.add(
        Conv2D(
            16,
            kernel_size=(nbr_channel, 1),
            padding="valid",
            strides=(1, 1),
            data_format="channels_first",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    # (16, 1, windows_len)
    # Pas trop de filtre pour éviter la redondance et le fait qu'un gros filtre se divise en plusieurs petits
    # Essayer de baisser le nombre de filtre (par exemple 8)
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
              padding="same", data_format="channels_first"))
    # (16, 1? windows_len/2)
    model.add(Dropout(0.5))
    # Convolution temporelle
    model.add(
        Conv2D(
            8,
            kernel_size=(1, 32),
            dilation_rate=(1, 2),
            data_format="channels_first",
            padding="same",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    # (8, 1, windows_len/2)
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same"))
    # (8, 1, windows_len/4)
    model.add(Dropout(0.5))
    model.add(
        Conv2D(
            4,
            kernel_size=(5, 5),
            dilation_rate=(2, 2),
            data_format="channels_first",
            padding="same",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    # (4, 1, windows_len/4)
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2),
              data_format="ch!annels_first", padding="same"))
    # (4, 1, windows_len/8)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(int(256), activation=None))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(2, name="preds", activation="softmax"))
    return model


# Créer un modèle de random forest avec une extraction des bandes de fréquences alpha, beta, ...
# A combiner avec le modèle de CNN