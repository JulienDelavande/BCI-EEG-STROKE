import numpy as np
import mne
from ml_eeg_tools.preprocessing import DataLoader
from ml_eeg_tools.preprocessing.data_manipulation import get_channels_side, replace_bad_channel_with_zeros


def prepare_data_inference(data_location, arm_side=None, brain_side=None, bad_channels=None, time_range=[0, None], fmin=1, fmax=40):
    """
    Prepare the data for the prediction.

    Parameters:
    ----------
        data_location (str): The location of the data.
        arm_side (str): The side of the arm ('D' for right and 'G' for left).
        brain_side (str): The side of the brain ('D' for right and 'G' for left).
        bad_channels (list): The list of bad channels to remove.
        time_range (list): The time range to keep.
        fmin (float): The minimum frequency for the filter.
        fmax (float): The maximum frequency for the filter.

    Returns:
    --------
        mne.io.Raw: The raw data modified.
    """

    data_loader = DataLoader(data_location)
    
    if brain_side is None:
        brain_side = data_loader.stroke_side
    if arm_side is None:
        arm_side = 'D' if data_loader.stroke_side == 'G' else 'G'

    # picking the arm session opposite to the stroke side
    raws = data_loader.get_raws(side=arm_side)

    # if no data for the arm side, we skip the session
    if raws is None:
        raise ValueError("No data for the arm side")

    # picking the channels of the selected brain side
    channels = get_channels_side(
        raws, brain_side
    )
    raws.pick_channels(channels)

    # Fill with zeros the channels to remove
    raws = replace_bad_channel_with_zeros(
        raws, bad_channels
    )

    # picking the time range
    start, end = time_range
    if end is None:
        end = raws.times[-1]
    elif end > raws.times[-1]:
        end = raws.times[-1]
    if start < raws.times[0]:
        start = raws.times[0]
    raws.crop(tmin=start, tmax=end)

    # filtering the data
    raws.filter(l_freq=fmin, h_freq=fmax, fir_design="firwin")

    return raws


def creating_sliding_windows(raws, window_size, window_step):
    """
    Create sliding windows from the EEG data.

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        window_size (float): The size of the window in seconds.
        window_step (float): The step of the window in seconds.

    Returns:
    --------
        np.array: The EEG data in windows.
    """
    picks = mne.pick_types(raws.info, eeg=True, stim=False)
    data = raws.get_data(picks=picks)
    sfreq = raws.info["sfreq"]
    window_size_idx = int(window_size * sfreq)
    window_step_idx = int(window_step * sfreq)
    # Découpage en fenêtres glissantes
    n_windows = (data.shape[1] - window_size_idx) // window_step_idx + 1
    window_data = []
    for i in range(n_windows):
        # Début/fin de la fenêtre
        start = i * window_step_idx
        end = start + window_size_idx + 1
        temp_window = data[:, start:end]
        window_data.append(temp_window)

    window_data = np.array(window_data)

    return window_data
