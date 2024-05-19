from sklearn.metrics import balanced_accuracy_score
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

"""
This module contains functions to train and test a pipeline on epochs data.
The goal is to compare the performance of the pipelines with cross-validation on different types test and train splits.
The cross-validation types are:
- A: Train and test on on same patients, same sessions (different epochs). 
- B: Train and test on on same patients, different sessions (different epochs).
- C: Train and test on different patients and different sessions.
- X: Train and test on all epoch data (patient, session, epoch shuffled) different epochs in test and train.
"""


def train_test_A(
    data_patients, labels_patients, pipeline, RANDOM_STATE=42, N_SPLIT=4, verbose=False
):
    """
    Train and test on on same patients, same sessions (different epochs).
    For each session of each patient, we split the epochs into train and test set for each fold.
    If N_SPLIT = 4, we have 4 folds, then in each fold 75% of the epochs are used for training and 25% for testing.
    We ensure different suffle for each fold by adding the fold number to the random state.

    Parameters
    ----------
    data_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of epochs,
        each epoch is a np.array of channels, each channel is a np.array of time samples
        (shape: n_patients*n_sessions*(n_epochs, n_channels, n_time_samples))
    labels_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of labels
        (shape: n_patients*n_sessions*(n_epochs))
    pipeline : sklearn.pipeline.Pipeline
        Pipeline to train and test
    RANDOM_STATE : int, optional
        Random state for shuffling the data. The default is 42.
    N_SPLIT : int, optional
        Number of splits for the cross-validation. The default is 4.
    verbose : bool, optional
        If True, print information about the scores. The default is False.

    Returns
    -------
    np.mean(scores) : float
        Mean of the balanced accuracy score for each fold.
    """

    # Creating folds

    # split_folds = [patient1, patient2, ...] dtype = list
    # patient1 = [session1_fold, session2_fold, ...] dtype = KFold

    # Examples of the folds (n split = 4) for the first patient, first session:

    # [(array([ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 12, 14, 16, 17, 18, 19, 20,
    #         21, 22, 23, 25, 28, 29, 31, 32, 34, 35, 37]),
    # array([ 4,  6, 13, 15, 24, 26, 27, 30, 33, 36])),
    # (array([ 1,  2,  3,  4,  6,  7, 10, 11, 13, 14, 15, 18, 20, 21, 22, 23, 24,
    #         26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37]),
    # array([ 0,  5,  8,  9, 12, 16, 17, 19, 25, 32])),
    # (array([ 0,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #         22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36]),
    # array([ 1,  2,  3, 11, 21, 29, 34, 35, 37])),
    # (array([ 0,  1,  2,  3,  4,  5,  6,  8,  9, 11, 12, 13, 15, 16, 17, 19, 21,
    #         24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37]),
    # array([ 7, 10, 14, 18, 20, 22, 23, 28, 31]))]

    # [(train_idx_array_epoch_fold_1, test_idx_array_epoch_fold_1),
    # (train_idx_array_epoch_fold_2, test_idx_array_epoch_fold_2), ...]

    split_folds = []
    i = 0
    for patient in range(len(data_patients)):
        split_folds.append([])
        for session in range(len(data_patients[patient])):
            random_state = RANDOM_STATE + i
            cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=random_state)
            split_folds[patient].append(cv.split(data_patients[patient][session]))
            i += 1

    # Training and testing
    scores = []
    for _ in range(N_SPLIT):
        X_train, y_train = [], []
        X_test, y_test = [], []
        for patient in range(len(data_patients)):
            for session in range(len(data_patients[patient])):
                train_index, test_index = next(split_folds[patient][session])
                X_train.append(data_patients[patient][session][train_index])
                y_train.append(labels_patients[patient][session][train_index])
                X_test.append(data_patients[patient][session][test_index])
                y_test.append(labels_patients[patient][session][test_index])
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))
        if verbose:
            print(f"balanced accuracy {scores[-1]}")

    if verbose:
        print(f"mean of balance accuracy {np.mean(scores)}")

    return np.mean(scores)


def train_test_B(
    data_patients, labels_patients, pipeline, RANDOM_STATE=42, N_SPLIT=4, verbose=False
):
    """
    Train and test on same patients, different sessions (different epochs).
    For each patient, we split the sessions into train and test set for each fold.
    Each patient has from 1 to 3 sessions so in each fold,sfor each patient, we have:
    - if 1 session: this session is used for training only, each fold contain this session for training and no session for testing.
    - if 2 sessions: one session is used for training and the other for testing, each fold contain one session for training and one for testing.
    - if 3 sessions: two sessions are used for training and the other for testing, each fold contain two sessions for training and one for testing.
    We ensure different suffle for each fold by adding the fold number to the random state.

    Parameters
    ----------
    data_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of epochs,
        each epoch is a np.array of channels, each channel is a np.array of time samples
        (shape: n_patients*n_sessions*(n_epochs, n_channels, n_time_samples))
    labels_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of labels
        (shape: n_patients*n_sessions*(n_epochs))
    pipeline : sklearn.pipeline.Pipeline
        Pipeline to train and test
    RANDOM_STATE : int, optional
        Random state for shuffling the data. The default is 42.
    N_SPLIT : int, optional
        Number of splits for the cross-validation. The default is 4.
    verbose : bool, optional
        If True, print information about the scores. The default is False.

    Returns
    -------
    np.mean(scores) : float
        Mean of the balanced accuracy score for each fold.
    """

    # Creating folds
    # split_folds = [patient1, patient2, ...] dtype = list
    # patient1 = [all_session_fold1, all_session_fold2, ...] dtype = tuple
    # all_session_fold1 = (train_idx, test_idx) dtype = list

    # Examples of the folds (n split = 4, n patient = 14):

    #  [[([1], [0]), ([1], [0]), ([1], [0]), ([0], [1])],
    #  [([0], [1]), ([0], [1]), ([1], [0]), ([1], [0])],
    #  [([1], [0]), ([1], [0]), ([0], [1]), ([0], [1])],
    #  [([0], [1]), ([0], [1]), ([0], [1]), ([0], [1])],
    #  [([0], [1]), ([0], [1]), ([0], [1]), ([0], [1])],
    #  [([1], [0]), ([1], [0]), ([1], [0]), ([1], [0])],
    #  [([1], [0]), ([0], [1]), ([0], [1]), ([1], [0])],
    #  [([1], [0]), ([0], [1]), ([1], [0]), ([1], [0])],
    #  [([0, 1], [2]), ([1, 2], [0]), ([0, 2], [1]), ([1, 2], [0])],
    #  [([0], [1]), ([0], [1]), ([0], [1]), ([0], [1])],
    #  [([0], [1]), ([1], [0]), ([1], [0]), ([1], [0])],
    #  [([1], [0]), ([1], [0]), ([1], [0]), ([0], [1])],
    #  [([0], [1]), ([1], [0]), ([1], [0]), ([0], [1])],
    #  [([0], []), ([0], []), ([0], []), ([0], [])]]

    # [(train_idx_array_session_fold_1_patient_1, test_idx_array_session_fold_1_patient_1),
    # (train_idx_array_session_fold_2_patient_1, test_idx_array_session_fold_2_patient_1), ...]

    split_folds = []
    i = 0
    for patient in range(len(data_patients)):
        split_folds.append([])
        for _ in range(N_SPLIT):
            if len(data_patients[patient]) < 2:
                test_idx = []
                train_idx = [0]
            else:
                shuffle_idx = np.arange(len(data_patients[patient]))
                np.random.RandomState(RANDOM_STATE + i).shuffle(shuffle_idx)
                test_idx = [shuffle_idx[-1]]
                train_idx = list(shuffle_idx[:-1])
            split_folds[-1].append((train_idx, test_idx))
            i += 1

    # Training and testing
    scores = []
    for fold in range(N_SPLIT):
        X_train, y_train = [], []
        X_test, y_test = [], []
        for patient in range(len(data_patients)):
            for session in split_folds[patient][fold][0]:
                X_train.append(data_patients[patient][session])
                y_train.append(labels_patients[patient][session])
            for session in split_folds[patient][fold][1]:
                X_test.append(data_patients[patient][session])
                y_test.append(labels_patients[patient][session])
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))
        if verbose:
            print(f"balanced accuracy {scores[-1]}")

    if verbose:
        print(f"mean of balance accuracy {np.mean(scores)}")

    return np.mean(scores)


def train_test_C(
    data_patients, labels_patients, pipeline, RANDOM_STATE=42, N_SPLIT=4, verbose=False
):
    """
    Train and test on different patients and different sessions.
    We split the patients into train and test set for each fold.
    If N_SPLIT = 4, we have 4 folds, then in each fold 75% of the patients are used for training and 25% for testing.

    Parameters
    ----------
    data_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of epochs,
        each epoch is a np.array of channels, each channel is a np.array of time samples
        (shape: n_patients*n_sessions*(n_epochs, n_channels, n_time_samples))
    labels_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of labels
        (shape: n_patients*n_sessions*(n_epochs))
    pipeline : sklearn.pipeline.Pipeline
        Pipeline to train and test
    RANDOM_STATE : int, optional
        Random state for shuffling the data. The default is 42.
    N_SPLIT : int, optional
        Number of splits for the cross-validation. The default is 4.
    verbose : bool, optional
        If True, print information about the scores. The default is False.

    Returns
    -------
    np.mean(scores) : float
        Mean of the balanced accuracy score for each fold.
    """

    # Creating folds
    # split_folds = [fold1, fold2, ...] dtype = list
    # fold1 = (train_idx, test_idx) dtype = list

    # Examples of the folds (n split = 4, n patient = 14):

    # [(array([ 1,  2,  3,  4,  5,  6,  7,  8, 10, 13]), array([ 0,  9, 11, 12])),
    #  (array([ 0,  3,  4,  6,  7,  9, 10, 11, 12, 13]), array([1, 2, 5, 8])),
    #  (array([ 0,  1,  2,  3,  5,  6,  8,  9, 10, 11, 12]), array([ 4,  7, 13])),
    #  (array([ 0,  1,  2,  4,  5,  7,  8,  9, 11, 12, 13]), array([ 3,  6, 10]))]

    # [(train_idx_array_patient_fold_1, test_idx_array_patient_fold_1),
    # (train_idx_array_patient_fold_2, test_idx_array_patient_fold_2), ...]

    cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    split_folds = cv.split(data_patients)
    scores = []
    i = 0
    for train_patient_idx, test_patient_idx in split_folds:
        train_data = np.concatenate(
            [
                data_patients[i][j]
                for i in train_patient_idx
                for j in range(len(data_patients[i]))
            ],
            axis=0,
        )
        train_labels = np.concatenate(
            [
                labels_patients[i][j]
                for i in train_patient_idx
                for j in range(len(data_patients[i]))
            ],
            axis=0,
        )
        test_data = np.concatenate(
            [
                data_patients[i][j]
                for i in test_patient_idx
                for j in range(len(data_patients[i]))
            ],
            axis=0,
        )
        test_labels = np.concatenate(
            [
                labels_patients[i][j]
                for i in test_patient_idx
                for j in range(len(data_patients[i]))
            ],
            axis=0,
        )

        # Shuffle train test
        train_idx = np.arange(len(train_data))
        np.random.RandomState(RANDOM_STATE + i).shuffle(train_idx)
        train_data = train_data[train_idx]
        train_labels = train_labels[train_idx]
        test_idx = np.arange(len(test_data))
        np.random.RandomState(RANDOM_STATE + i).shuffle(test_idx)
        test_data = test_data[test_idx]
        test_labels = test_labels[test_idx]
        i += 1

        # Train
        pipeline.fit(train_data, train_labels)
        y_pred = pipeline.predict(test_data)
        scores.append(balanced_accuracy_score(test_labels, y_pred))
        if verbose:
            print(f"balanced accuracy {scores[-1]}")

    if verbose:
        print(f"mean of balance accuracy {np.mean(scores)}")

    return np.mean(scores)


def train_test_X(
    data_patients, labels_patients, pipeline, RANDOM_STATE=42, N_SPLIT=4, verbose=False
):
    """
    Train and test on all epoch data (patient, session, epoch shuffled) different epochs in test and train.
    We concatenate all the epochs of all the patients and sessions and shuffle them.
    Then we split the epochs into train and test set for each fold.
    If N_SPLIT = 4, we have 4 folds, then in each fold 75% of the epochs are used for training and 25% for testing.

    Parameters
    ----------
    data_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of epochs,
        each epoch is a np.array of channels, each channel is a np.array of time samples
        (shape: n_patients*n_sessions*(n_epochs, n_channels, n_time_samples))
    labels_patients : list
        List patients, each patient is a list of sessions, each session is a np.array of labels

    pipeline : sklearn.pipeline.Pipeline
        Pipeline to train and test
    RANDOM_STATE : int, optional
        Random state for shuffling the data. The default is 42.
    N_SPLIT : int, optional
        Number of splits for the cross-validation. The default is 4.
    verbose : bool, optional
        If True, print information about the scores. The default is False.

    Returns
    -------
    np.mean(scores) : float
        Mean of the balanced accuracy score for each fold.
    """
    epochs_data_collapsed = np.concatenate(
        [np.concatenate(data_patients[i], axis=0) for i in range(len(data_patients))],
        axis=0,
    )
    labels_collapsed = np.concatenate(
        [np.concatenate(labels_patients[i], axis=0) for i in range(len(data_patients))],
        axis=0,
    )
    cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline,
        epochs_data_collapsed,
        labels_collapsed,
        cv=cv,
        scoring="balanced_accuracy",
    )
    if verbose:
        print(f"mean of balance accuracy {np.mean(scores)}")
    return np.mean(scores)
