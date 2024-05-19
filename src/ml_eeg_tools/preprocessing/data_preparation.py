import tqdm
import numpy as np
import mne
from ml_eeg_tools.preprocessing.data_manipulation import get_channels
from ml_eeg_tools.preprocessing import DataLoader

def prepare_data_train(file_path_list: str, settings: dict, verbose=False):
    """
     Prepare data for training.
     Both for movement and movement intention classification.
     Possible to use the data for binary classification (movement vs no movement)
     or for multiclass classification (flexion vs extension vs no movement).

     The function is composed of the following steps:
     1. Load the data for each file
     2. Select the sessions within the file that correspond to the arm in oposition to the stroke side (if not, skip the session)
     3. Select the channels of interest which are the channels of the stroke side (and the ones in the middle line -> z)
     4. Apply the bandpass filter
     5. Create the epochs for the movement and no movement (and movement intention if provided) with labels
     6. Shuffle the data
     7. Return the data

     Parameters
     ----------
     file_path_list : list
         List of file paths (.npy files containing the eeg data)
     settings : dict
         Dictionary of settings
         {
             'FMIN': int, # minimum frequency for the bandpass filter
             'FMAX': int, # maximum frequency for the bandpass filter
             'EPOCHS_TMIN': float, # start time of the epoch (from movement onset)
             'EPOCHS_TMAX': float, # end time of the epoch  (from movement onset)
             'EPOCHS_EMPTY_FROM_MVT_TMIN': float, # start time of the epoch for no movement (from movement onset)
             'BINARY_CLASSIFICATION': bool, # if True, the labels are 0 and 1 (0 for no movement, 1 for movement)
             else the labels are 0, 1, 2 (0 for no movement, 1 for flexion, 2 for extension)
             'RANDOM_STATE': int, # random state for shuffling the data
             'EPOCHS_INTENTION_FROM_MVT_TMIN': float (optional) # start time of the epoch for movement intention
             (from movement onset) if not provided, no movement intention epochs are created
         }
     verbose : bool
         If True, print information about the data

     Returns
     -------
     data_patients : list
         List patients, each patient is a list of sessions, each session is a np.array of epochs,
         each epoch is a np.array of channels, each channel is a np.array of time samples
         (shape: n_patients*n_sessions*(n_epochs, n_channels, n_time_samples))
     labels_patients : list
         List patients, each patient is a list of sessions, each session is a np.array of labels
         (shape: n_patients*n_sessions*(n_epochs))
     patients_id : list
         List of patient ids (str)
         (shape: n_patients)
     sessions_id : list
         List of patients, each patient is a list of sessions ids (str)
         (shape: n_patients*(n_sessions))

    For more information on how to use it, check the related notebook in the notebooks folder
    (notebooks/ml_eeg_tools_example/data_preparation.ipynb) .

    """

    # Structure of the data output

    # structure data_patients = [session_1, session_2, ...]            dtype = list
    # structure session_i     = [epoch_1, epoch_2, ...]                dtype = np.array
    # structure epoch_i       = np.array([channel_1, channel_2, ...])  dtype = np.array
    # structure channel_i     = np.array([time_1, time_2, ...])        dtype = np.float64

    # structure labels_patients     = [session_1, session_2, ...]            dtype = list
    # structure session_i           = [label_epoch_1, label_epoch_2, ...]    dtype = np.float64 (0, 1, 2)

    data_patients = []
    labels_patients = []

    patients_id = []
    sessions_id = []

    # Loop over the files
    for file in tqdm.tqdm(file_path_list):

        # loading data / auto labeling behind the scene
        data_loader = DataLoader(file)

        # picking the arm session opposite to the stroke side
        stroke = data_loader.stroke_side
        arm_side = "G" if stroke == "D" else "G"
        raws = data_loader.get_raws(side=arm_side)

        # if no data for the arm side, we skip the session
        if raws is None:
            continue

        # picking channels of only the stroke side
        raws.pick_channels(get_channels(raws, stroke))

        # filtering
        raws.filter(settings["FMIN"], settings["FMAX"], fir_design="firwin")

        # creating epochs over flexion and extension of the arm (movement)
        events = mne.find_events(raws, stim_channel=["movement"])
        picks = mne.pick_types(raws.info, eeg=True, stim=False)
        epochs = mne.Epochs(
            raws,
            events,
            tmin=settings["EPOCHS_TMIN"],
            tmax=settings["EPOCHS_TMAX"],
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs_data_mvt = epochs.get_data()
        label_mvt = epochs.events[:, -1]
        if "EPOCHS_INTENTION_FROM_MVT_TMIN" in settings:
            label_mvt = np.zeros(epochs_data_mvt.shape[0])

        # creating artificial epoch when no movement
        epochs_no_mvt = mne.Epochs(
            raws,
            events,
            tmin=settings["EPOCHS_EMPTY_FROM_MVT_TMIN"],
            tmax=settings["EPOCHS_EMPTY_FROM_MVT_TMIN"]
            + settings["EPOCHS_TMAX"]
            - settings["EPOCHS_TMIN"],
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs_data_no_mvt = epochs_no_mvt.get_data()
        labels_no_mvt = np.zeros(epochs_data_no_mvt.shape[0])

        # crating epochs before movement (movement intention)
        if "EPOCHS_INTENTION_FROM_MVT_TMIN" in settings:
            epochs_mvnt_intention = mne.Epochs(
                raws,
                events,
                tmin=settings["EPOCHS_INTENTION_FROM_MVT_TMIN"],
                tmax=settings["EPOCHS_INTENTION_FROM_MVT_TMIN"]
                + settings["EPOCHS_TMAX"]
                - settings["EPOCHS_TMIN"],
                picks=picks,
                baseline=None,
                preload=True,
            )
            epochs_data_mvnt_intention = epochs_mvnt_intention.get_data()
            labels_mvnt_intention = np.ones(epochs_data_mvnt_intention.shape[0])

        # concatenating epochs
        epochs_data_session = np.concatenate(
            (epochs_data_mvt, epochs_data_no_mvt), axis=0
        )
        labels_session = np.concatenate((label_mvt, labels_no_mvt), axis=0)

        if "EPOCHS_INTENTION_FROM_MVT_TMIN" in settings:
            epochs_data_session = np.concatenate(
                (epochs_data_session, epochs_data_mvnt_intention), axis=0
            )
            labels_session = np.concatenate(
                (labels_session, labels_mvnt_intention), axis=0
            )

        # Shuffling epochs
        random_state = np.random.RandomState(settings["RANDOM_STATE"])
        indices = np.arange(epochs_data_session.shape[0])
        random_state.shuffle(indices)
        epochs_data_session = epochs_data_session[indices]
        labels_session = labels_session[indices]

        # if binary classification, we merge flexion and extension
        if settings["BINARY_CLASSIFICATION"]:
            labels_session[labels_session != 0] = 1

        # adding session data and session labels to data_patients and labels_patients (if new patient, add new list)
        if data_loader.patient_id not in patients_id:
            patients_id.append(data_loader.patient_id)
            sessions_id.append([])
            data_patients.append([])
            labels_patients.append([])
        sessions_id[patients_id.index(data_loader.patient_id)].append(
            data_loader.session_id
        )
        data_patients[patients_id.index(data_loader.patient_id)].append(
            epochs_data_session
        )
        labels_patients[patients_id.index(data_loader.patient_id)].append(
            labels_session
        )

        if verbose:
            print(f"patient id: {data_loader.patient_id}")
            print(f"session id: {data_loader.session_id}")
            print(f"number of epochs: {epochs_data_session.shape[0]}")
            print(f"number of channels: {epochs_data_session.shape[1]}")
            print(f"number of time samples: {epochs_data_session.shape[2]}")

    return data_patients, labels_patients, patients_id, sessions_id
