"""
Classif module providing functions for training a DL/ML classification 
model. 

List of classes & functions:
    Classes:
        - EarlyStopping

    Functions:
        - train_one_epoch
        - valid_one_epoch
        - train_model
"""

import numpy as np
import torch

from classif.utils import save_model_dict, load_model_dict


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a
    given patience.

    Attributes:
        - patience: How long to wait after last time validation loss
                    improved.
                    Default: 5
        - delta   : Minimum change in the monitored quantity to qualify
                    as an improvement.
                    Default: 0
    """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.best_score = None
        self.stop_early = False

    def step(self, loss):
        score = -loss
        if self.best_score is not None and \
           self.best_score + self.delta > score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop_early = True
        else:
            self.best_score = score
            self.counter = 0


def train_one_epoch(
        model,
        optimizer,
        criterion,
        train_loader,
        torch_DEVICE,
        verbose=True
):
    """
    Function to train a model for one epoch.

    :param model       : The model to train.
    :param optimizer   : The optimizer to use (SGD, Adam, ...).
    :param criterion   : The loss function to use (MSE, BCE, ...).
    :param train_loader: The training data loader.
    :param torch_DEVICE: The device to use (CPU, GPU, ...).
    :param verbose     : Whether to print statistics during training or not.

    :return: The average loss over the epoch.
    """
    epoch_loss = []

    for i, batch in enumerate(train_loader):
        x, y_true = batch
        x = x.to(torch_DEVICE)

        # format the y_true so that it is compatible with the loss
        y_true = y_true.to(torch_DEVICE)
        y_true = y_true.view((-1, 1)).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_pred = model(x)

        # compute loss and backpropagate
        loss = criterion(y_pred, y_true)
        loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        epoch_loss.append(loss.item())

        if verbose and i % 2 == 0:
            print(f"Batch {i:5d}, curr loss = {loss.item():.03f}")

    return np.asarray(epoch_loss).mean()


def valid_one_epoch(
        model,
        optimizer,
        criterion,
        valid_loader,
        torch_DEVICE,
        verbose=True
):
    """
    Function to validate a model for one epoch.

    :param model       : The model to validate.
    :param optimizer   : The optimizer to use (SGD, Adam, ...). [not used]
    :param criterion   : The loss function to use (MSE, BCE, ...).
    :param valid_loader: The validation data loader.
    :param torch_DEVICE: The device to use (CPU, GPU, ...).
    :param verbose     : Whether to print statistics during validation or not.

    :return: The average loss over the epoch.
    """
    epoch_loss = []

    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            x, y_true = batch
            x = x.to(torch_DEVICE)

            # format the y_true so that it is compatible with the loss
            y_true = y_true.to(torch_DEVICE)
            y_true = y_true.view((-1, 1)).float()

            # forward
            y_pred = model(x)

            # compute loss
            loss = criterion(y_pred, y_true)

            # save statistics
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


def train_model(
    model,
    optimizer,
    criterion,
    n_epochs,
    train_loader,
    valid_loader,
    torch_DEVICE,
    early_stop=True,
    early_cntr=5,
    checkpoint=True,
    model_path="model.pt",
):
    """
    Function to complete a full training of a model.

    :param model       : The model to train.
    :param optimizer   : The optimizer to use (SGD, Adam, ...).
    :param criterion   : The loss function to use (MSE, BCE, ...).
    :param n_epochs    : The number of epochs to train for.
    :param train_loader: The training data loader.
    :param valid_loader: The validation data loader.
    :param torch_DEVICE: The device to use (CPU, GPU, ...).
    :param early_stop  : Whether to use early stopping or not.
    :param checkpoint  : Whether to save the best model or not.
    :param model_path  : The path to save the model to (if checkpoint=True).

    :return: The training and validation losses over the epochs.
    """
    train_losses = []
    valid_losses = []

    if checkpoint:
        path = model_path
    if early_stop:
        estop = EarlyStopping(patience=early_cntr)

    for epoch in range(n_epochs):
        model.train()
        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            torch_DEVICE)

        model.eval()
        valid_epoch_loss = valid_one_epoch(
            model,
            optimizer,
            criterion,
            valid_loader,
            torch_DEVICE)

        print(
            f"EPOCH={epoch}, TRAIN={train_epoch_loss}, VALID={valid_epoch_loss}")

        train_losses.append(train_epoch_loss)
        valid_losses.append(valid_epoch_loss)

        # Add early stopping
        if early_stop:
            estop.step(valid_epoch_loss)
        if early_stop and estop.stop_early:
            break

        # Add best backup checkpoint
        if checkpoint and valid_epoch_loss <= np.min(valid_losses):
            save_model_dict(model, path)

    if checkpoint:
        load_model_dict(model, path)

    return train_losses, valid_losses
