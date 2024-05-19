import numpy as np
from schemas import schemas
from services import data_manipulation_service
from services.data_loader import DataLoader
import mne


def prepare_data(user_selection, pipeline):
    """
    Prepare the data for the prediction.

    Parameters:
    ----------
        user_selection (schemas.UserSelection): The user selection.
        pipeline (dict): The pipeline information.

    Returns:
    --------
        mne.io.Raw: The raw data.
        schemas.ProcessingInfo: The processing information.
    """

    data_loader = DataLoader(user_selection.data_location)

    # picking the arm session opposite to the stroke side
    raws = data_loader.get_raws(side=user_selection.arm_side)

    # if no data for the arm side, we skip the session
    if raws is None:
        raise ValueError("No data for the arm side")

    # picking the channels of the selected brain side
    channels = data_manipulation_service.get_channels_side(
        raws, user_selection.brain_side
    )
    raws.pick_channels(channels)

    # Fill with zeros the channels to remove
    raws = data_manipulation_service.replace_bad_channel_with_zeros(
        raws, user_selection.bad_channels
    )

    # picking the time range
    start, end = user_selection.time_range # [0, None] by default
    if end is None:
        end = raws.times[-1]
    elif end > raws.times[-1]:
        end = raws.times[-1]
    if start < raws.times[0]:
        start = raws.times[0]
    raws.crop(tmin=start, tmax=end)

    # filtering the data
    raws.filter(l_freq=pipeline["FMIN"], h_freq=pipeline["FMAX"], fir_design="firwin")

    processing_info = schemas.ProcessingInfo(
        patient_id=data_loader.patient_id,
        stroke_side=data_loader.stroke_side,
        session_id=data_loader.session_id,
        arm_side=user_selection.arm_side,
        electrodes=channels,
        time_range=(start, end),
        model=user_selection.model,
        data_location=user_selection.data_location,
        pipeline_parameters=pipeline,
    )

    return raws, processing_info


def creating_sliding_windows(raws, pipeline):
    """
    Create sliding windows from the EEG data.

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        pipeline (dict): The pipeline information.

    Returns:
    --------
        np.array: The EEG data in windows.
    """
    picks = mne.pick_types(raws.info, eeg=True, stim=False)
    data = raws.get_data(picks=picks)
    sfreq = raws.info["sfreq"]
    window_size_idx = int(pipeline["WINDOWS_SIZE"] * sfreq)
    window_step_idx = int(pipeline["WINDOWS_STEP"] * sfreq)
    # Découpage en fenêtres glissantes
    n_windows = (data.shape[1] - window_size_idx) // window_step_idx + 1
    window_data = []
    for i in range(n_windows):
        # Début/fin de la fenêtre
        start = i * window_step_idx
        end = start + window_size_idx
        temp_window = data[:, start:end]
        window_data.append(temp_window)

    window_data = np.array(window_data)

    return window_data
