# Train and test the model
# python 3.8.8
import os
import pandas as pd
import time
import numpy as np
import psutil
from joblib import dump
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from ml_eeg_tools.preprocessing.data_preparation import prepare_data_train
from ml_eeg_tools.test_model.test_epoch import train_test_A, train_test_B, train_test_C, train_test_X


### HYPERPARAMETERS ############################################################
RANDOM_STATE = 42
FMIN = 9
FMAX = 30
EPOCHS_TMIN = -0.75
EPOCHS_LENGTH = 1.7
EPOCHS_TMAX = EPOCHS_TMIN + EPOCHS_LENGTH
EPOCHS_EMPTY_FROM_MVT_TMINS = -6

cov = Covariances(estimator='oas')
ts = TangentSpace()
cl = SVC(random_state=RANDOM_STATE)
pipeline = Pipeline([('Cov', cov), ('TangentSpace', ts), ('Classifier', cl)])


### CONFIGURATION ##############################################################
PATH_RESULTS = './results/'
PATH_MODEL = './models/'
NAME_MODEL = 'RIEMANN_SVM_MVT_v2_no_1st_session'
NAME_RESULTS = f'{NAME_MODEL}_scores'
BINARY_CLASSIFICATION = True
N_SPLIT = 4
FOLDER_PATH = './../../../data/raw/Data_npy/'
FILE_PATH_LIST = [FOLDER_PATH + file_path for file_path in os.listdir(FOLDER_PATH) if file_path.endswith('.npy')]
NUMBER_OF_SESSIONS = 85
SAVE_PREPROCESSED_DATA = True
USE_SAVED_PREPROCESSED_DATA = False

### SCRIPT #####################################################################
def run_experiment():
    duration = time.time()
    settings = {
        'FMIN': FMIN,
        'FMAX': FMAX,
        'EPOCHS_TMIN': EPOCHS_TMIN,
        'EPOCHS_TMAX': EPOCHS_TMAX,
        'EPOCHS_EMPTY_FROM_MVT_TMIN': EPOCHS_EMPTY_FROM_MVT_TMINS,
        'BINARY_CLASSIFICATION': BINARY_CLASSIFICATION,
        'RANDOM_STATE': RANDOM_STATE
    }
    data_patients, labels_patients, _, _ = prepare_data_train(FILE_PATH_LIST[1:], settings)
    print(f"Data loaded: {len(data_patients)} patients, {len(labels_patients)} labels")

    file_path = PATH_RESULTS + 'data_patients_labels.npy'
    # Save data
    if SAVE_PREPROCESSED_DATA:
        
        with open(file_path, 'wb') as f:
            np.save(f, {
                'data_patients': data_patients,
                'labels_patients': labels_patients,

        })
        print(f"Data saved: {file_path}")

    # Load data
    if USE_SAVED_PREPROCESSED_DATA:
        with open(file_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            data_patients = data.item().get('data_patients')
            labels_patients = data.item().get('labels_patients')

    # Train and test the model
    score_A = train_test_A(data_patients, labels_patients, pipeline, RANDOM_STATE=RANDOM_STATE, N_SPLIT=N_SPLIT, verbose=False)
    print(f"Score A: {score_A}")
    score_B = train_test_B(data_patients, labels_patients, pipeline, RANDOM_STATE=RANDOM_STATE, N_SPLIT=N_SPLIT, verbose=False)
    print(f"Score B: {score_B}")
    score_C = train_test_C(data_patients, labels_patients, pipeline, RANDOM_STATE=RANDOM_STATE, N_SPLIT=N_SPLIT, verbose=False)
    print(f"Score C: {score_C}")
    score_X = train_test_X(data_patients, labels_patients, pipeline, RANDOM_STATE=RANDOM_STATE, N_SPLIT=N_SPLIT, verbose=False)
    print(f"Score X: {score_X}")

    # Some more info about the process
    duration = time.time() - duration # in seconds
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024 # in MB

    data_patients_collapsed = np.concatenate([np.concatenate(data_patients[i], axis=0) for i in range(len(data_patients))], axis=0)
    labels_patients_collapsed = np.concatenate([np.concatenate(labels_patients[i], axis=0) for i in range(len(data_patients))], axis=0)

    # fit and save model on all data
    pipeline.fit(data_patients_collapsed, labels_patients_collapsed)
    accuracy_train = pipeline.score(data_patients_collapsed, labels_patients_collapsed)
    date = time.strftime("%Y-%m-%d")
    model_info = {
    'pipeline': pipeline,
    'additionnal_info': {
        'aim': 'Binary classification of EEG signals to detect movement',
        'date_of_training': date,
        'data_preprocessing_details': 'StandardScaler, Covariances with OAS estimator, TangentSpace mapping',
        'pipeline_parameters': pipeline.get_params(),
        'preprocessing_parameters': {
            'FMIN': FMIN,
            'FMAX': FMAX,
            'EPOCHS_TMIN': EPOCHS_TMIN,
            'EPOCHS_TMAX': EPOCHS_TMAX,
            'EPOCHS_EMPTY_FROM_MVT_TMINS': EPOCHS_EMPTY_FROM_MVT_TMINS,
            'BINARY_CLASSIFICATION': BINARY_CLASSIFICATION,
            'RANDOM_STATE': RANDOM_STATE,
            'N_SPLIT': N_SPLIT,
            'NUMBER_OF_SESSIONS': NUMBER_OF_SESSIONS
        },
        'model_performance': {
            'score_A': score_A,
            'score_B': score_B,
            'score_C': score_C,
            'score_X': score_X,
            'accuracy_train': accuracy_train,
        }
    }
    }
    dump(model_info, PATH_MODEL + NAME_MODEL + f'_{date}.joblib')

    # Save results in a csv file
    line = {
        'EPOCHS_TMIN': EPOCHS_TMIN, 
        'EPOCHS_LENGTH': EPOCHS_LENGTH, 
        'EPOCHS_EMPTY_FROM_MVT_TMINS': EPOCHS_EMPTY_FROM_MVT_TMINS, 
        'FMIN': FMIN, 'FMAX': FMAX,
        'CLASSIFIER': 'SVC', 
        'RANDOM_STATE': RANDOM_STATE, 'N_SPLIT': N_SPLIT, 
        'NUMBER_OF_SESSIONS': NUMBER_OF_SESSIONS, 
        'score_A': score_A, 'score_B': score_B, 'score_C': score_C, 'score_X': score_X, 
        'duration': duration,
        'memory': memory,
    }
    results = pd.DataFrame([line])
    results.to_csv(PATH_RESULTS + NAME_RESULTS + f'_{date}.csv', index=False)
    return line

# Create directory for results, models
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)

# Train and test
line  = run_experiment()
print(f"Saved: {line} to {PATH_RESULTS + NAME_RESULTS}.csv and {PATH_MODEL + NAME_MODEL}.joblib")
