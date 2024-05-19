import mne
import numpy as np

HEAD_SIDE_RIGHT = "D"


def get_channels_side(raws, side, indices=False):
    """
    Get the channels of the specified side.

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        side (str): The side of the head ('D' for right and 'G' for left).
        indices (bool): Whether to return the indices of the channels or the names.

    Returns:
    --------
        list: The channels of the specified side.
    """
    endings = (
        ("1", "3", "5", "7", "9")
        if side == HEAD_SIDE_RIGHT
        else ("2", "4", "6", "8", "10")
    )
    channels_to_remove = [
        channel for channel in raws.ch_names if channel.endswith(endings)
    ]
    channels = [
        channel for channel in raws.ch_names if channel not in channels_to_remove
    ]
    if indices:
        channels = [raws.ch_names.index(channel) for channel in channels]
    return channels


def get_channels(raws, side=None, electrodes=None):
    """
    Get the EEG and stimulus channels of the specified side.

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        side (str): The side of the head ('D' for right and 'G' for left).
        electrodes (list): The list of electrodes to include.

    Returns:
    --------
        tuple: The EEG and stimulus channels.
    """
    if electrodes is None:
        picks_eeg = mne.pick_types(raws.info, meg=False, eeg=True, stim=False)
        picks_stim = mne.pick_types(raws.info, meg=False, eeg=False, stim=True)
    else:
        picks_eeg = mne.pick_channels(
            raws.info["ch_names"], include=electrodes, exclude="stim"
        )
        picks_stim = mne.pick_channels(raws.info["ch_names"], include=["stim"])

    if side is not None:
        channels_eeg = get_channels_side(raws, side, indices=True)
        picks_eeg = list(set(channels_eeg) & set(picks_eeg))

    return picks_eeg, picks_stim


def replace_bad_channel_with_zeros(raws, electrodes_to_remove):
    if electrodes_to_remove is not None:
        for electrode in electrodes_to_remove:
            if electrode in raws.ch_names:
                raws._data[raws.ch_names.index(electrode)] = np.zeros(
                    raws._data[0].shape
                )
    return raws
