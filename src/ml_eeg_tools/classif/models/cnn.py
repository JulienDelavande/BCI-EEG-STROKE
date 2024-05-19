"""
Module to define CNN models.
"""

import torch
import torch.nn as nn


class CNN_Windows(nn.Module):
    """
    Base architecture CNN model. Used for the sliding window approach.

    Args:
        nb_channels (int): Number of channels in the input data.
        windows_len (int): Size of the window in the input data.

    Note:
        The CNN architecture can be decomposed as follows:

        - a convolution layer with 16 filters of size (nb_channels, 1) used as 
          a spatial filter stage,
        - a convolution layer with 8 filters of size (1, 32) used as a temporal
          filter stage,
        - a convolution layer with 4 filters of size (1, 5) used as a temporal
          filter stage too,
        - a flatten layer.   
    """

    def __init__(
            self,
            nb_channels: int,
            windows_len: int) -> None:
        super(CNN_Windows, self).__init__()

        self.net = nn.Sequential(
            # Data shape = ( 1, nb_channels, windows_len)
            nn.Conv2d(
                1,
                16,
                kernel_size=(nb_channels, 1),
                padding="valid"
            ),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.2),

            # Data shape = (16, 1, windows_len/2)
            nn.Conv2d(
                16,
                8,
                kernel_size=(1, 32),
                dilation=(1, 2),
                padding="same"
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.2),

            # Data shape = ( 8, 1, windows_len/4)
            nn.Conv2d(
                8,
                4,
                kernel_size=(1, 5),
                dilation=(1, 2),
                padding="same"
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.2),

            # Data shape = ( 4, 1, windows_len/8)
            nn.Flatten(),
            nn.Linear(4 * (windows_len // 8), 256),
            nn.LeakyReLU(0.3),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        return self.net(x)
