import torch.jit
import torch
import numpy as np
import tqdm
from typing import Callable
from torch.utils.data import DataLoader, Dataset



# Helper function to use datasets
class NpArrayDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        windows_transforms: Callable = None,
        targets_transforms: Callable = None,
    ):
        self.windows = windows
        self.targets = targets
        self.windows_transforms = windows_transforms
        self.targets_transforms = targets_transforms

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index: int):
        x = self.windows[index]
        y = self.targets[index]

        if self.windows_transforms is not None:
            x = self.windows_transforms(x)
        else:
            x = torch.tensor(x)

        if self.targets_transforms is not None:
            y = self.targets_transforms(y)
        else:
            y = torch.tensor(y)

        return x, y

def save_model_dict(model, model_path):
    """
    Function to save a model's state dict to a file.

    :param model     : The model to save.
    :param model_path: The path to save the model to.
    """
    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)


def save_model_full(model, model_path):
    """
    Function to save a model to a file using torch.jit.

    :param model     : The model to save.
    :param model_path: The path to save the model to.
    """
    model = model.cpu().eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(model_path)


def load_model_dict(model, model_path):
    """
    Function to load a model's state dict from a file.

    :param model     : The model to load the state dict into.
    :param model_path: The path to load the model from.
    """
    model.load_state_dict(torch.load(model_path))


def load_model_full(model_path, torch_DEVICE):
    """
    Function to load a model from a file using torch.jit.

    :param model_path  : The path to load the model from.
    :param torch_DEVICE: The device to use (CPU, GPU, ...).

    :return: The loaded model.
    """
    model = torch.jit.load(model_path, map_location=torch_DEVICE)
    return model


def predict_test(
        model,
        data,
        torch_DEVICE):
    """
    Function to predict the labels of a dataset.

    :param model       : The model to use for prediction.
    :param data        : The dataset to predict on.
    :param torch_DEVICE: The device to use (CPU, GPU, ...).

    NB: The dataset given as :data: must be a torch.utils.data.Dataset 
    and must contain both inputs and labels for the samples.
    """
    y_true = []
    y_pred = []

    model.to(torch_DEVICE)
    model.eval()

    with torch.no_grad():
        for x, y_t in tqdm.tqdm(data, "Predicting"):
            x = x.reshape((-1,) + x.shape)
            x = x.to(torch_DEVICE)

            y_p = model.forward(x)
            y_p = y_p.to("cpu").numpy()
            y_t = y_t.to("cpu").numpy().astype(int)

            y_true.append(y_t)
            y_pred.append(y_p)

        y_true = np.asarray(y_true)
        y_pred = np.concatenate(y_pred, axis=0)

    return y_pred, y_true
