from joblib import load
import numpy as np
from models.pipelines import pipelines

PATH_MODELS = "./models/"
PREDICTION_PIPELINE_MODEL_KEY = "pipeline"
PREDICTION_PIPELINE_INFO_KEY = "additional_info"


def predict(windows, processing_info):
    """
    Predict the movement based on the EEG data.

    Parameters:
    ----------
        windows (np.array): The EEG data in windows.
        processing_info (schemas.ProcessingInfo): The processing information.

    Returns:
    --------
        np.array: The predictions.
        dict: The pipeline information.
    """
    # Charger le modèle
    model_name = processing_info.model  # RIEMANN_SVM_MVT_V1
    model_name_path = pipelines[model_name]["pipeline_name"]
    model_info_loaded = load(PATH_MODELS + model_name_path)
    pipeline_loaded = model_info_loaded[PREDICTION_PIPELINE_MODEL_KEY]
    # pipeline_info = model_info_loaded[PREDICTION_PIPELINE_INFO_KEY]
    pipeline_info = processing_info.pipeline_parameters

    y_pred = pipeline_loaded.predict(windows)

    return y_pred, pipeline_info


def density_on_prediction(raws, predictions, pipeline):
    """
    Reconstruct the prediction on the temporal signal (prediction density).

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        predictions (np.array): The predictions.
        pipeline (dict): The pipeline information.

    Returns:
    --------
        list: The density of the prediction.
    """
    # Tableau de densité de prédiction
    time = raws.times
    window_size_idx = int(pipeline["WINDOWS_SIZE"] * raws.info["sfreq"])
    window_step_idx = int(pipeline["WINDOWS_STEP"] * raws.info["sfreq"])

    n_windows = len(predictions)
    density = np.zeros(len(time))
    count = np.zeros(len(time))
    for i in range(n_windows):
        start = i * window_step_idx
        end = start + window_size_idx
        density[start:end] += predictions[i]  # /(size/steps)
        count[start:end] += 1
    # Normalisation

    density = [
        density[i] / count[i] if count[i] != 0 else 0 for i in range(len(density))
    ]

    return density
