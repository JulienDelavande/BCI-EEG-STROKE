import numpy as np
import mne
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

# Note:
#      - Ne fonctionne que sur la structure des fichiers numpy générés à partir des fichiers .mat -> script data_extraction.py
#      - Ne fonctionne que si les 4 signaux cinématiques sont présents ['AC', 'VAC', 'AC3d', 'VAC3d']


class DataLoader:
    """
    Class to load the data from the .npy file (created through the .mat files thanks to the data_extraction.py script)
    and create the RawArray objects.
    Each .npy file contains two recordings, one for each arm. Sometimes one is missing.

    Args:
    -----
    file_path : str
        The path to the .npy file.
    mne_log_level : str
        The log level for MNE. Default is 'ERROR'.

    Attributes:
    -----------
    file_path : str
        The path to the .npy file.
    data : np.array
        The data loaded from the .npy file.
    patient_id : str
        The patient id.
    stroke_side : str
        The side of the stroke.
    session_id : str
        The session id.
    cinematic_freq : int
        The frequency of the cinematic signals.
    eeg_freq : int
        The frequency of the eeg signals.
    signal_left_arm : dict
        The eeg signals for the left arm. The keys are the electrodes and the values are the signals.
    signal_right_arm : dict
        The eeg signals for the right arm. The keys are the electrodes and the values are the signals.
    movement_index_left_arm : np.array
        The movement index for the left arm. Each value is the index of a movement in the cinematic signals.
    movement_index_right_arm : np.array
        The movement index for the right arm. Each value is the index of a movement in the cinematic signals.
    cinematic_signals_left_arm : dict
        The cinematic signals for the left arm. The keys are the channels and the values are the signals.
    cinematic_signals_right_arm : dict
        The cinematic signals for the right arm. The keys are the channels and the values are the signals.
    right_arm_electrodes : np.array
        The electrodes for the right arm.
    left_arm_electrodes : np.array
        The electrodes for the left arm.
    signal_right_arm_raws : mne.io.RawArray
        The RawArray object for the right arm. It has all the eeg signals, the cinematic signals and the labels.
        - EEG signals: the electrodes are the channels.
        - Cinematic signals: the channels are 'AC', 'VAC', 'AC3d', 'VAC3d'.
        - Labels: the channels are 'movement' ('movement' is created from the 'AC' signal, a notebook is available
        in the notebooks folder to understand how it is created) and 'extension' ('extension' is the label provided by
        the CHU).
        All the signals are resampled to fit the eeg frequency.
    signal_left_arm_raws : mne.io.RawArray
        The RawArray object for the left arm. It has all the eeg signals, the cinematic signals and the labels.
        - EEG signals: the electrodes are the channels.
        - Cinematic signals: the channels are 'AC', 'VAC', 'AC3d', 'VAC3d'.
        - Labels: the channels are 'movement' ('movement' is created from the 'AC' signal, a notebook is available
        in the notebooks folder to understand how it is created) and 'extension' ('extension' is the label provided by
        the CHU).
        All the signals are resampled to fit the eeg frequency.
    close_label_detection_treshold_time : int
        The threshold time to detect close labels (used in the creation of the labels 'movement').
    """

    def __init__(self, file_path: str, mne_log_level: str = "ERROR") -> None:

        # MNE log level
        mne.set_log_level(mne_log_level)

        # File path
        self.file_path = file_path
        self.data = None
        self._load_data()

        # Meta data
        self.patient_id: str = self.data[0]
        self.stroke_side: str = self.data[1]
        self.session_id: str = self.data[2]
        self.cinematic_freq: int = int(self.data[3])
        self.eeg_freq: int = 1024

        # Data
        self.signal_left_arm: dict = {}
        self.signal_right_arm: dict = {}
        self.movement_index_left_arm: np.array = np.array([], dtype=int)
        self.movement_index_right_arm: np.array = np.array([], dtype=int)
        self.cinematic_signals_left_arm: dict = {}
        self.cinematic_signals_right_arm: dict = {}

        # Electrodes
        self.right_arm_electrodes: np.array = np.array([], dtype=str)
        self.left_arm_electrodes: np.array = np.array([], dtype=str)
        self.cinematic_channels = ["AC", "VAC", "AC3d", "VAC3d"]

        self._fill_signals()

        # Raws objects
        self.signal_right_arm_raws: mne.io.RawArray = None
        self.signal_left_arm_raws: mne.io.RawArray = None

        self.close_label_detection_treshold_time = 5  # seconds
        self._fill_raws()

        # useful size

        self.signal_right_arm_duration: float = (
            self.signal_right_arm_raws.n_times / self.eeg_freq
            if self.signal_right_arm_raws is not None
            else 0
        )
        self.signal_left_arm_duration: float = (
            self.signal_left_arm_raws.n_times / self.eeg_freq
            if self.signal_left_arm_raws is not None
            else 0
        )

        self.signal_right_arm_first_movement_time = (
            self.movement_index_right_arm[0] / self.cinematic_freq
            if len(self.movement_index_right_arm) > 0
            else 0
        )
        self.signal_left_arm_first_movement_time = (
            self.movement_index_left_arm[0] / self.cinematic_freq
            if len(self.movement_index_left_arm) > 0
            else 0
        )
        self.signal_right_arm_last_movement_time = (
            self.movement_index_right_arm[-1] / self.cinematic_freq
            if len(self.movement_index_right_arm) > 0
            else 0
        )
        self.signal_left_arm_last_movement_time = (
            self.movement_index_left_arm[-1] / self.cinematic_freq
            if len(self.movement_index_left_arm) > 0
            else 0
        )

    def __str__(self) -> str:
        return f"DataLoader(session_id={self.session_id}, stroke_side={self.stroke_side}, session_id={self.session_id}, \n \
            signal_right_arm_duration={self.signal_right_arm_duration}s, signal_left_arm_duration={self.signal_left_arm_duration}s, \n \
            signal_right_arm_first_movement_time={self.signal_right_arm_first_movement_time}s, signal_left_arm_first_movement_time={self.signal_left_arm_first_movement_time}s, \n \
            signal_right_arm_last_movement_time={self.signal_right_arm_last_movement_time}s, signal_left_arm_last_movement_time={self.signal_left_arm_last_movement_time}s, \n \
            signal_right_arm_size={self.signal_right_arm_raws.n_times}, signal_left_arm_size={self.signal_left_arm_raws.n_times}, \n \
            movement_index_right_arm_size={len(self.movement_index_right_arm)}, movement_index_left_arm_size={len(self.movement_index_left_arm)}, \n \
            right_arm_electrodes_size={len(self.right_arm_electrodes)}, left_arm_electrodes_size={len(self.left_arm_electrodes)})"

    def _load_data(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            self.data = np.load(self.file_path, allow_pickle=True)
        except Exception as e:
            raise IOError(f"Error loading the file: {str(e)}")

    def _fill_signals(self) -> None:
        micro_to_sec = 1e-6
        for i in range(4, len(self.data)):
            if type(self.data[i]) == list:
                if self.data[i][0] == "D":
                    self.movement_index_right_arm = np.array(
                        [int(self.data[i][1][j]) for j in range(len(self.data[i][1]))],
                        dtype=int,
                    )
                    self.signal_right_arm = {
                        self.data[i][2][j][0]: self.data[i][2][j][1] * micro_to_sec
                        for j in range(len(self.data[i][2]))
                    }
                    self.cinematic_signals_right_arm = {
                        self.cinematic_channels[j]: self.data[i][3][j]
                        for j in range(len(self.data[i][3]))
                    }
                    self.right_arm_electrodes = np.array(
                        [self.data[i][2][j][0] for j in range(len(self.data[i][2]))],
                        dtype=str,
                    )
                if self.data[i][0] == "G":
                    self.movement_index_left_arm = np.array(
                        [int(self.data[i][1][j]) for j in range(len(self.data[i][1]))],
                        dtype=int,
                    )
                    self.signal_left_arm = {
                        self.data[i][2][j][0]: self.data[i][2][j][1] * micro_to_sec
                        for j in range(len(self.data[i][2]))
                    }
                    self.cinematic_signals_left_arm = {
                        self.cinematic_channels[j]: self.data[i][3][j]
                        for j in range(len(self.data[i][3]))
                    }
                    self.left_arm_electrodes = np.array(
                        [self.data[i][2][j][0] for j in range(len(self.data[i][2]))],
                        dtype=str,
                    )

    def _fill_raws(self) -> None:

        montage = mne.channels.make_standard_montage("standard_1020")

        ###############################################################################################################
        if len(self.signal_left_arm) > 0:

            # Labels, corresponding to the RawArray
            cinematic_signals_left_arm_nsample: int = self.cinematic_signals_left_arm[
                list(self.cinematic_signals_left_arm.keys())[0]
            ].shape[0]
            label_left_arm: np.array = np.zeros(
                cinematic_signals_left_arm_nsample, dtype=int
            )
            label_left_arm[self.movement_index_left_arm] = 1
            label_left_arm_resampled = interp1d(
                np.arange(len(label_left_arm)), label_left_arm, kind="nearest"
            )(
                np.linspace(
                    0,
                    len(label_left_arm) - 1,
                    len(list(self.signal_left_arm.values())[0]),
                )
            ).astype(
                int
            )

            # Resampling the cinematic signals to fit the eeg frequency
            cinematic_signals_left_arm_resampled = {
                self.cinematic_channels[j]: interp1d(
                    np.arange(
                        len(self.cinematic_signals_left_arm[self.cinematic_channels[j]])
                    ),
                    self.cinematic_signals_left_arm[self.cinematic_channels[j]],
                    kind="nearest",
                )(
                    np.linspace(
                        0,
                        len(self.cinematic_signals_left_arm[self.cinematic_channels[j]])
                        - 1,
                        len(list(self.signal_left_arm.values())[0]),
                    )
                )
                for j in range(len(self.cinematic_channels))
            }

            # Creating labels for flexion (2) and extension (1) of the wrist, no movement (0)
            label_left_arm_flexion_extension = fill_with_last_non_nan(
                cinematic_signals_left_arm_resampled["AC"]
            )
            label_left_arm_flexion_extension = acc_to_movement(
                label_left_arm_flexion_extension
            )
            label_left_arm_flexion_extension = remove_close_labels_all(
                label_left_arm_flexion_extension,
                threshold_step=self.close_label_detection_treshold_time * self.eeg_freq,
            )

            # Creating the RawArray
            all_data_left_arm = np.array(
                list(self.signal_left_arm.values())
                + list(cinematic_signals_left_arm_resampled.values())
                + [label_left_arm_flexion_extension]
                + [label_left_arm_resampled]
            )
            ch_names_left_arm = (
                list(self.left_arm_electrodes)
                + self.cinematic_channels
                + ["movement"]
                + ["extension"]
            )
            ch_types_left_arm = (
                ["eeg"] * len(self.left_arm_electrodes)
                + ["stim"] * len(self.cinematic_channels)
                + ["stim"] * 2
            )
            info_left_arm = mne.create_info(
                ch_names_left_arm, self.eeg_freq, ch_types_left_arm
            )
            self.signal_left_arm_raws: mne.io.RawArray = mne.io.RawArray(
                all_data_left_arm, info_left_arm
            )
            self.signal_left_arm_raws.set_montage(montage)

        else:
            self.signal_left_arm_raws = None

            ###############################################################################################################

        if len(self.signal_right_arm) > 0:

            # Labels, corresponding to the RawArray
            cinematic_signals_right_arm_nsample: int = self.cinematic_signals_right_arm[
                list(self.cinematic_signals_right_arm.keys())[0]
            ].shape[0]

            label_right_arm: np.array = np.zeros(
                cinematic_signals_right_arm_nsample, dtype=int
            )
            label_right_arm[self.movement_index_right_arm] = 1

            label_right_arm_resampled = interp1d(
                np.arange(len(label_right_arm)), label_right_arm, kind="nearest"
            )(
                np.linspace(
                    0,
                    len(label_right_arm) - 1,
                    len(list(self.signal_right_arm.values())[0]),
                )
            ).astype(
                int
            )

            # Resampling the cinematic signals to fit the eeg frequency
            cinematic_signals_right_arm_resampled = {
                self.cinematic_channels[j]: interp1d(
                    np.arange(
                        len(
                            self.cinematic_signals_right_arm[self.cinematic_channels[j]]
                        )
                    ),
                    self.cinematic_signals_right_arm[self.cinematic_channels[j]],
                    kind="nearest",
                )(
                    np.linspace(
                        0,
                        len(
                            self.cinematic_signals_right_arm[self.cinematic_channels[j]]
                        )
                        - 1,
                        len(list(self.signal_right_arm.values())[0]),
                    )
                )
                for j in range(len(self.cinematic_channels))
            }

            # Creating labels for flexion (2) and extension (1) of the wrist, no movement (0)
            label_right_arm_flexion_extension = fill_with_last_non_nan(
                cinematic_signals_right_arm_resampled["AC"]
            )
            label_right_arm_flexion_extension = acc_to_movement(
                label_right_arm_flexion_extension
            )
            label_right_arm_flexion_extension = remove_close_labels_all(
                label_right_arm_flexion_extension,
                threshold_step=self.close_label_detection_treshold_time * self.eeg_freq,
            )

            # Creating the RawArray
            all_data_right_arm = np.array(
                list(self.signal_right_arm.values())
                + list(cinematic_signals_right_arm_resampled.values())
                + [label_right_arm_flexion_extension]
                + [label_right_arm_resampled]
            )
            ch_names_right_arm = (
                list(self.right_arm_electrodes)
                + self.cinematic_channels
                + ["movement"]
                + ["extension"]
            )
            ch_types_right_arm = (
                ["eeg"] * len(self.right_arm_electrodes)
                + ["stim"] * len(self.cinematic_channels)
                + ["stim"] * 2
            )
            info_right_arm = mne.create_info(
                ch_names_right_arm, self.eeg_freq, ch_types_right_arm
            )
            self.signal_right_arm_raws: mne.io.RawArray = mne.io.RawArray(
                all_data_right_arm, info_right_arm
            )
            self.signal_right_arm_raws.set_montage(montage)

        else:
            self.signal_right_arm_raws = None

        ###############################################################################################################

    def get_raws(self, side: str) -> mne.io.RawArray:
        if side == "G":
            return self.signal_left_arm_raws
        elif side == "D":
            return self.signal_right_arm_raws
        else:
            raise ValueError(f"Wrong side: {side}, should be 'G' or 'D'")


# Functions used to process the accelerometer data and extract the labels


# 1. Filling missing values with the last non-nan value
def fill_with_last_non_nan(array):
    last_valid = None
    for i in range(len(array)):
        if not np.isnan(array[i]):
            last_valid = array[i]
        else:
            array[i] = last_valid
    return array


# 2. Smoothing the signal and labeling the movements
def acc_to_movement(acc):
    new_acc_smoothed = np.zeros(acc.shape)
    med = np.mean(acc)
    new_acc_smoothed[acc > med] = 1
    movement_new = np.zeros(new_acc_smoothed.shape)
    movement_new[1:] = new_acc_smoothed[:-1]
    movement_new = movement_new - new_acc_smoothed
    movement_new[0] = 0
    movement_new[movement_new == -1] = 2
    return movement_new


# 3. Removing the labels that are too close to each other
def remove_close_labels(data, threshold_step=5 * 1024, reverse=True):
    if reverse:
        data = data[::-1]
    for i in range(len(data)):
        if data[i : i + threshold_step].sum() > 2:
            data[i : i + threshold_step] = 0
    if reverse:
        data = data[::-1]
    return data


def remove_close_labels_all(data, threshold_step=5 * 1024):
    data_reverse = data[::-1]
    for i in range(len(data)):
        if data[i : i + threshold_step].sum() > 2:
            data[i : i + threshold_step] = 0
    for i in range(len(data_reverse)):
        if data_reverse[i : i + threshold_step].sum() > 2:
            data_reverse[i : i + threshold_step] = 0
    data_reverse = data_reverse[::-1]
    data_sum = data - data_reverse
    data_to_remove = data_sum != 0
    data[data_to_remove] = 0

    return data


if __name__ == "__main__":

    mne.set_log_level("ERROR")

    FILE_PATH = "./data/raw/Data_npy/DATA_001_Trial1.npy"
    loader = DataLoader(FILE_PATH)
    print(loader)

    raws = loader.get_raws("G")
    print(raws)

    raws.plot()
    raws.plot_psd()
    raws.plot_sensors("3d", ch_type="eeg")
    raws.plot_sensors("3d", ch_type="eeg", show_names=True)
    plt.show()
