# This file contains the pipelines used in the application. Each pipeline is a dictionary with the following keys:
# - pipeline_name: The name of the pipeline file
# - FMIN: The minimum frequency
# - FMAX: The maximum frequency
# - WINDOWS_SIZE: The size of the window
# - WINDOWS_STEP: The step of the window
# - type: The type of the pipeline (movement_prediction, intention_prediction)
# - date: The date of the pipeline
# - scores: The scores of the pipeline
# - composition: The composition of the pipeline
# - info_train: The information about the training of the pipeline
#
# The pipelines are used in the application to predict the movement and intention of the patient.

pipelines = {
    "RIEMANN_SVM_MVT_V1": {
        "pipeline_name": "RIEMANN_SVM_MVT_v1_2024-03-01.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_prediction",
        "date": "2024-03-01",
        "scores": {
            "score_A": 0.855,
            "score_B": 0.840,
            "score_C": 0.757,
            "score_X": 0.855,
            "score_meanABC": 0.817,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 89,
        },
        "composition": "Covariances -> TangentSpace -> StandardScaler -> SVC",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side",
    },
        "RIEMANN_SVM_MVT_V2": {
        "pipeline_name": "RIEMANN_SVM_MVT_v2_2024-03-08.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_prediction",
        "date": "2024-03-08",
        "scores": {
            "score_A": 0.884,
            "score_B": 0.856,
            "score_C": 0.735,
            "score_X": 0.884,
            "score_meanABC": 0.825,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 89,
        },
        "composition": "Covariances -> TangentSpace -> SVC",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side",
    },
        "RIEMANN_MLP_INT_MVT_V1": {
        "pipeline_name": "RIEMANN_MLP_INT_MVT_v1_2024-03-08.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_intention_prediction",
        "date": "2024-03-08",
        "scores": {
            "score_A": 0.598,
            "score_B": 0.601,
            "score_C": 0.543,
            "score_X": 0.579,
            "score_meanABC": 0.581,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 89,
        },
        "composition": "Covariances -> TangentSpace -> MLPClassifier",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side",
    },
    "RIEMANN_MLP_INT_MVT_V1_0-85sessions": {
        "pipeline_name": "RIEMANN_MLP_INT_MVT_v1_0-85sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_intention_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.598,
            "score_B": 0.601,
            "score_C": 0.543,
            "score_X": 0.579,
            "score_meanABC": 0.581,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 85,
        },
        "composition": "Covariances -> TangentSpace -> MLPClassifier",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 85 first sessions",
    },
    "RIEMANN_MLP_INT_MVT_V1_4-90sessions": {
        "pipeline_name": "RIEMANN_MLP_INT_MVT_v1_4-90sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_intention_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.598,
            "score_B": 0.601,
            "score_C": 0.543,
            "score_X": 0.579,
            "score_meanABC": 0.581,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 85,
        },
        "composition": "Covariances -> TangentSpace -> MLPClassifier",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 4-90th sessions",
    },
    "RIEMANN_MLP_INT_MVT_V1_1-90sessions": {
        "pipeline_name": "RIEMANN_MLP_INT_MVT_v1_1-90sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_intention_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.598,
            "score_B": 0.601,
            "score_C": 0.543,
            "score_X": 0.579,
            "score_meanABC": 0.581,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 88,
        },
        "composition": "Covariances -> TangentSpace -> MLPClassifier",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 1-90th sessions",
    },
     "RIEMANN_SVM_MVT_V2_0-85sessions": {
        "pipeline_name": "RIEMANN_SVM_MVT_v2_0-85sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.884,
            "score_B": 0.856,
            "score_C": 0.735,
            "score_X": 0.884,
            "score_meanABC": 0.825,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 85,
        },
        "composition": "Covariances -> TangentSpace -> SVC",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 85 first sessions",
    },
    "RIEMANN_SVM_MVT_V2_1-90sessions": {
        "pipeline_name": "RIEMANN_SVM_MVT_v2_1-90sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.884,
            "score_B": 0.856,
            "score_C": 0.735,
            "score_X": 0.884,
            "score_meanABC": 0.825,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 88,
        },
        "composition": "Covariances -> TangentSpace -> SVC",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 1-90th sessions",
    },
    "RIEMANN_SVM_MVT_V2_4-90sessions": {
        "pipeline_name": "RIEMANN_SVM_MVT_v2_4-90sessions_2024-03-10.joblib",
        "FMIN": 9,
        "FMAX": 30,
        "WINDOWS_SIZE": 1.7,
        "WINDOWS_STEP": 0.2,
        "type": "movement_prediction",
        "date": "2024-03-10",
        "scores": {
            "score_A": 0.884,
            "score_B": 0.856,
            "score_C": 0.735,
            "score_X": 0.884,
            "score_meanABC": 0.825,
            "N_SPLIT": 4,
            "RANDOM_STATE": 42,
            "NUMBER_OF_SESSIONS": 85,
        },
        "composition": "Covariances -> TangentSpace -> SVC",
        "info_train": "trained on sessions with arm opposing the stroke side and electrode of the stroke side 4-90th sessions",
    },
}

if __name__ == "__main__":

    from joblib import load

    # load the model
    pipeline_name = "RIEMANN_SVM_MVT_v2_85_first_sessions"
    pipeline = load(pipelines[pipeline_name]["pipeline_name"])
    print(pipeline)
