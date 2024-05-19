
import mne
import numpy as np

HEAD_SIDE_RIGHT = "D"

def get_channels(raws, side):
    """
    Get the channels of the stroke side

    Parameters
    ----------
    raws : mne.io.Raw
        Raw data
    side : str
        Side of the stroke (D for right, G for left) or the side to keep.

    Returns
    -------
    list
        List of channels of the stroke side (side to keep) and the middle line (z)
    """
    endings = ("1", "3", "5", "7", "9") if side == "D" else ("2", "4", "6", "8", "10")
    channels_to_remove = [
        channel for channel in raws.ch_names if channel.endswith(endings)
    ]
    channels = [
        channel for channel in raws.ch_names if channel not in channels_to_remove
    ]
    return channels


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


def replace_bad_channel_with_zeros(raws, electrodes_to_remove):
    if electrodes_to_remove is not None:
        for electrode in electrodes_to_remove:
            if electrode in raws.ch_names:
                raws._data[raws.ch_names.index(electrode)] = np.zeros(
                    raws._data[0].shape
                )
    return raws
