import os
import pandas as pd
import time
import uuid
import psutil
from sklearn.model_selection import ParameterGrid
from ml_eeg_tools.preprocessing.data_preparation import prepare_data_train
from ml_eeg_tools.test_model.test_epoch import (
    train_test_A,
    train_test_B,
    train_test_C,
    train_test_X,
)
import argparse
from joblib import Parallel, delayed
import pandas as pd
import time
import numpy as np


def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description="Grid search hyperparameters.")
    parser.add_argument(
        "--node-index", type=int, required=True, help="Index of the current node."
    )
    parser.add_argument(
        "--total-nodes", type=int, required=True, help="Total number of nodes."
    )
    parser.add_argument(
        "--version", type=int, required=True, help="Version of the grid search."
    )
    parser.add_argument(
        "--cores-per-node", type=str, required=True, help="Number of cores per node."
    )
    return parser.parse_args()


def prepare_data_one_core(settings, params, file_id, node_index=1):
    """
    Prepare the data (preprocessing) to be done on one core.
    It saves the data for the specified parameters in a .npy file. in the folder [PATH_DATA_PROCESSED]/NODE_{node_index}/.

    Parameters
    ----------
    settings : dict
        Dictionary containing the settings. Has to contain the following keys:
        settings = {
            'FILE_PATH': str, path to the data files
            'PATH_DATA_PROCESSED': str, path to the processed data
            'BINARY_CLASSIFICATION': bool, binary classification if True, ternary classification if False
            'RANDOM_STATE': int, random state
            'VERSION': int, version of the grid search
            'SAVE_EACH_ITERATION': bool, save each iteration if True
            'PATH_RESULTS': str, path to the results
            'NAME_RESULTS_ITERATIONS': str, name of the results for each iteration
            'FOLDER_RESULTS_ITERATIONS': str, folder for the results of each iteration
            'REMOVE_ITERATIONS': bool, remove the iterations if True
            'REMOVE_DATA_PROCESSED': bool, remove the data processed if True
            'LOG_FOLDER': str, path to the log folder
        }
    params : dict
        Dictionary containing the parameters. Has to contain the following keys:
        params = {
            'FMIN': int, minimum frequency
            'FMAX': int, maximum frequency
            'EPOCHS_TMIN': int, start time for the epochs
            'EPOCHS_TMAX': int, end time for the epochs (if not specified, the length of the epochs, EPOCHS_LENGTH is used)
            'EPOCHS_EMPTY_FROM_MVT_TMIN': int, start time for the epochs empty from movement
            'EPOCHS_LENGTH': int, length of the epochs (if not specified, EPOCHS_TMAX is used)
            'EPOCHS_INTENTION_FROM_MVT_TMIN': int (optional), start time for the epochs intention from movement
        }
    file_id : str
        ID of the file.
    node_index : int
        Index of the node.

    Returns
    -------
    str
        File path.
    """
    start_time = time.time()

    # Prepare the data
    settings_prepare_data = {
        "FMIN": params["FMIN"],
        "FMAX": params["FMAX"],
        "EPOCHS_TMIN": params["EPOCHS_TMIN"],
        "EPOCHS_TMAX": (
            params["EPOCHS_TMIN"] + params["EPOCHS_LENGTH"]
            if "EPOCHS_LENGTH" in params
            else params["EPOCHS_TMAX"]
        ),
        "EPOCHS_EMPTY_FROM_MVT_TMIN": params["EPOCHS_EMPTY_FROM_MVT_TMIN"],
        "BINARY_CLASSIFICATION": settings["BINARY_CLASSIFICATION"],
        "RANDOM_STATE": settings["RANDOM_STATE"],
    }
    if "EPOCHS_INTENTION_FROM_MVT_TMIN" in params:
        settings_prepare_data["EPOCHS_INTENTION_FROM_MVT_TMIN"] = params[
            "EPOCHS_INTENTION_FROM_MVT_TMIN"
        ]

    file_path_list = [
        settings["FILE_PATH"] + file_path
        for file_path in os.listdir(settings["FILE_PATH"])
        if file_path.endswith(".npy")
    ]

    try:
        data_patients, labels_patients, _, _ = prepare_data_train(
            file_path_list[: settings["NUMBER_OF_SESSIONS"]], settings_prepare_data
        )
    except Exception as e:
        print(f"Error: {e}")
        print(f"params: {params}")
        return None

    end_time = time.time()

    # Save the data
    path_data_processed_node = (
        settings["PATH_DATA_PROCESSED"] + "/" + "NODE_" + str(node_index) + "/"
    )
    if not os.path.exists(path_data_processed_node):
        os.makedirs(path_data_processed_node)

    file_path = path_data_processed_node + file_id + ".npy"
    with open(file_path, "wb") as f:
        np.save(
            f,
            {
                "data_patients": data_patients,
                "labels_patients": labels_patients,
                "duration": end_time - start_time,
                "params": params,
            },
        )
    return file_path


def train_test_one_core(settings, pipeline_dict, node_index=1):
    """
    Train and test the model for one core.
    It saves the results in a csv file for each iteration if SAVE_EACH_ITERATION is True.
    Saves the results in the folder {PATH_RESULTS}/V{VERSION}/{FOLDER_RESULTS_ITERATIONS}.

    Parameters
    ----------
    settings : dict
        Dictionary containing the settings. Has to contain the following keys:
        settings = {
            'RANDOM_STATE': int, random state
            'N_SPLIT': int, number of splits for the cross-validation
            'SAVE_EACH_ITERATION': bool, save each iteration if True
            'PATH_RESULTS': str, path to the results
            'NAME_RESULTS_ITERATIONS': str, name of the results for each iteration
            'FOLDER_RESULTS_ITERATIONS': str, folder for the results of each iteration
            'REMOVE_ITERATIONS': bool, remove the iterations if True
            'LOG_FOLDER': str, path to the log folder
        }
    pipeline_dict : dict
        Dictionary containing the pipeline and the parameters. Has to contain the following
        keys:
        pipeline_dict = {
            'file_preprocessing_path': str, path to the file containing the preprocessed data
            'pipeline': pipeline,
            'param_1': value_1,
            ...
        }
    node_index : int
        Index of the node.

    Returns
    -------
    line : dict
        Results of the training and testing.
        line = {
            'param_1': value_1,
            ...
            'score_A': float, score A
            'score_B': float, score B
            'score_C': float, score C
            'score_X': float, score X
            'duration_preprocessing': float, duration of the preprocessing
            'duration_train': float, duration of the training
            'memory': float, memory used
            'RANDOM_STATE': int, random state
            'N_SPLIT': int, number of splits for the cross-validation
            'SAVE_EACH_ITERATION': bool, save each iteration if True
            'PATH_RESULTS': str, path to the results
            'NAME_RESULTS_ITERATIONS': str, name of the results for each iteration
            'FOLDER_RESULTS_ITERATIONS': str, folder for the results of each iteration
            'REMOVE_ITERATIONS': bool, remove the iterations if True
            'LOG_FOLDER': str, path to the log folder
        }
    """

    start_time = time.time()

    # Open the correct file
    with open(pipeline_dict["file_preprocessing_path"], "rb") as f:
        data = np.load(f, allow_pickle=True).item()
        data_patients = data["data_patients"]
        labels_patients = data["labels_patients"]
        duration_preprocessing = data["duration"]
        params = data["params"]

    # Prepare the pipeline
    pipeline_params = {
        key: value
        for key, value in pipeline_dict.items()
        if key != "file_preprocessing_path" and key != "pipeline"
    }
    pipeline = pipeline_dict["pipeline"]
    print(f"pipeline_params: {pipeline_params}")
    pipeline.set_params(**pipeline_params)
    print(f"pipeline: {pipeline}")

    # Train and test the model
    try:
        score_A = train_test_A(
            data_patients,
            labels_patients,
            pipeline,
            RANDOM_STATE=settings["RANDOM_STATE"],
            N_SPLIT=settings["N_SPLIT"],
            verbose=False,
        )
        score_B = train_test_B(
            data_patients,
            labels_patients,
            pipeline,
            RANDOM_STATE=settings["RANDOM_STATE"],
            N_SPLIT=settings["N_SPLIT"],
            verbose=False,
        )
        score_C = train_test_C(
            data_patients,
            labels_patients,
            pipeline,
            RANDOM_STATE=settings["RANDOM_STATE"],
            N_SPLIT=settings["N_SPLIT"],
            verbose=False,
        )
        score_X = train_test_X(
            data_patients,
            labels_patients,
            pipeline,
            RANDOM_STATE=settings["RANDOM_STATE"],
            N_SPLIT=settings["N_SPLIT"],
            verbose=False,
        )
    except Exception as e:
        print(f"Error: {e}")
        print(f"pipeline_params: {pipeline_params}")
        return None

    # Some more info about the process
    duration_train = time.time() - start_time  # in seconds
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB

    # Save results in a csv file for each iteration
    line = {
        **params,
        "pipeline": pipeline,
        **pipeline_params,
        "score_A": score_A,
        "score_B": score_B,
        "score_C": score_C,
        "score_X": score_X,
        "duration_preprocessing": duration_preprocessing,
        "duration_train": duration_train,
        "memory": memory,
        **settings,
    }

    if settings["SAVE_EACH_ITERATION"]:
        results = pd.DataFrame([line])
        # Create the folder if it does not exist
        path_results_iterations = (
            settings["PATH_RESULTS"]
            + f'/V{settings["VERSION"]}'
            + settings["FOLDER_RESULTS_ITERATIONS"]
        )
        if not os.path.exists(path_results_iterations):
            os.makedirs(path_results_iterations)
        # Save the results
        results.to_csv(
            path_results_iterations
            + settings["NAME_RESULTS_ITERATIONS"]
            + f'_V{settings["VERSION"]}'
            + f"_node_{node_index}_{uuid.uuid1()}.csv",
            index=False,
        )

    return line


def run_search_one_node(
    settings,
    params_dict_lists,
    pipelines_dict_lists,
    params_dict_lists_exclude=None,
    pipelines_dict_lists_exclude=None,
    params_exclude_rules=None,
    pipelines_exclude_rules=None,
    node_index=1,
    total_nodes=1,
    total_cores=6,
    estimation_time_it=660,
):
    """
    Run the grid search for one node.
    It prepares the data and then trains and tests the model for each parameter in the grid.
    1. It uses multicore processing for the preprocessing and saves the results in a .npy file for a set preproc paramters.
    2. When it is done, it uses multicore processing for the training and testing on a set of pipeline parameters and saves the results in a csv file for each iteration if SAVE_EACH_ITERATION is True.
    3. When it's finished it saves the results in the folder {PATH_RESULTS}/V{VERSION}.
    It also create a log file for each node in the folder {LOG_FOLDER}/V{VERSION}.

    Parameters
    ----------
    settings : dict
        Dictionary containing the settings. Has to contain the following keys:
        settings = {
            'FILE_PATH': str, path to the data files
            'PATH_DATA_PROCESSED': str, path to the processed data
            'BINARY_CLASSIFICATION': bool, binary classification if True, ternary classification if False
            'RANDOM_STATE': int, random state
            'VERSION': int, version of the grid search
            'SAVE_EACH_ITERATION': bool, save each iteration if True
            'PATH_RESULTS': str, path to the results
            'NAME_RESULTS': str, name of the results
            'FOLDER_RESULTS_ITERATIONS': str, folder for the results of each iteration
            'REMOVE_ITERATIONS': bool, remove the iterations if True
            'REMOVE_DATA_PROCESSED': bool, remove the data processed if True
            'LOG_FOLDER': str, path to the log folder
        }
    params_dict_lists : dict
        Dictionary containing the parameters for the preprocessing. Has to contain the following keys:
        params_dict_lists = {
            'FMIN': list, list of minimum frequencies
            'FMAX': list, list of maximum frequencies
            'EPOCHS_TMIN': list, list of start times for the epochs
            'EPOCHS_TMAX': list, list of end times for the epochs (if not specified, the length of the epochs, EPOCHS_LENGTH is used)
            'EPOCHS_EMPTY_FROM_MVT_TMIN': list, list of start times for the epochs empty from movement
            'EPOCHS_LENGTH': list, list of lengths of the epochs (if not specified, EPOCHS_TMAX is used)
            'EPOCHS_INTENTION_FROM_MVT_TMIN': list (optional), list of start times for the epochs intention from movement
        }
    pipelines_dict_lists : dict
        Dictionary containing the pipelines and the parameters. Has to contain the following keys:
        pipelines_dict_lists = {
            'pipeline_1': {
                'pipeline': pipeline_1,
                'param_1': list_1,
                ...
            },
            'pipeline_2': {
                'pipeline': pipeline_2,
                'param_1': list_1,
                ...
            },
            ...
        }
    params_dict_lists_exclude : dict
        Dictionary containing the parameters to exclude for the preprocessing. Has to contain the same keys as params_dict_lists.
    pipelines_dict_lists_exclude : dict
        Dictionary containing the parameters to exclude for the pipelines. Has to contain the same keys as pipelines_dict_lists.
    params_exclude_rules : list
        List of rules to exclude some parameters for the preprocessing.
    pipelines_exclude_rules : list
        List of rules to exclude some parameters for the pipelines.
    node_index : int
        Index of the node. (Used for knowing which parameters from params_dict_lists and pipelines_dict_lists to use on this very node, with total nodes)
    total_nodes : int
        Total number of nodes. (Used for knowing which parameters from params_dict_lists and pipelines_dict_lists to use on this very node, with node index)
    total_cores : int
        Number of cores per node. (Used for estimating the time of the preprocessing and the training and testing)
    estimation_time_it : float
        Estimated time for one iteration in seconds.
    """

    node_index = int(node_index)
    total_nodes = int(total_nodes)
    total_cores = int(total_cores)
    estimation_time_it = float(estimation_time_it)

    # Create the grid
    grid_preprocessing = ParameterGrid(params_dict_lists)
    print(f"grid_preprocessing before exclude: {grid_preprocessing}")

    # Exclude some parameters
    if params_dict_lists_exclude is not None:
        grid_preprocessing_exclude = ParameterGrid(params_dict_lists_exclude)
        grid_preprocessing = [
            params
            for params in grid_preprocessing
            if params not in grid_preprocessing_exclude
        ]
    print(f"grid_preprocessing after exclude: {grid_preprocessing}")
    if params_exclude_rules is not None:
        for rule in params_exclude_rules:
            grid_preprocessing = [
                params for params in grid_preprocessing if not rule(params)
            ]
    print(f"grid_preprocessing after rules: {grid_preprocessing}")

    # Get the grid for the current node
    # If the numbers of configs in the preprocessing is greater than the number of nodes, we split the grid on the nodes
    # Else we perform the same preprocessing on every node
    if (
        len(grid_preprocessing) >= total_nodes
    ):  # We share the preprocessing accross the nodes
        grid_preprocessing = list(grid_preprocessing)[node_index - 1 :: total_nodes]

    # Calculate some predictions for each node
    node_log_path = (
        settings["LOG_FOLDER"]
        + f'V{settings["VERSION"]}/'
        + settings["NAME_RESULTS"]
        + f'_V{settings["VERSION"]}_node{node_index}_log.txt'
    )
    if not os.path.exists(settings["LOG_FOLDER"] + f'V{settings["VERSION"]}/'):
        os.makedirs(settings["LOG_FOLDER"] + f'V{settings["VERSION"]}/')
    with open(node_log_path, "w") as file:
        file.write(
            f'### LOG {settings["NAME_RESULTS"]} V{settings["VERSION"]} ############################\n'
        )
        file.write(f'Name of the results: {settings["NAME_RESULTS"]}\n')
        file.write(f'Version: {settings["VERSION"]}\n')
        file.write(f"Node index: {node_index}\n")
        file.write(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        file.write(f"### DEVICE ############################\n")
        file.write(f"Number of nodes: {total_nodes}\n")
        file.write(f"Number of cores: {total_cores}\n\n")

        file.write(f"### Parameters preprocessing ############################\n")
        file.write(f"Number of parameters for this node: {len(grid_preprocessing)}\n")
        file.write(
            f"Number of param per core: {len(grid_preprocessing)//total_cores}\n"
        )
        file.write(
            f"Estimated time for this node: {(len(grid_preprocessing)//total_cores) * estimation_time_it/60} minutes, {(len(grid_preprocessing)//total_cores) * estimation_time_it/3600} hours\n\n"
        )

        file.write(f"### Grid preprocessing ############################\n")
        file.write(f"params: {params_dict_lists}\n\n")

    # Run the the preprocessing for the current node
    n_jobs = -1
    start_time_node_preprocessing = time.time()
    path_file_list = Parallel(n_jobs=n_jobs)(
        delayed(prepare_data_one_core)(settings, params, f"params_{i}", node_index)
        for i, params in enumerate(grid_preprocessing)
    )
    duration_node_preprocessing = time.time() - start_time_node_preprocessing

    # Deal with the None values
    path_file_list = [path for path in path_file_list if path is not None]

    # Save the time of the preprocessing
    with open(node_log_path, "a") as file:
        file.write(f"### Results preprocessing ############################\n")
        file.write(
            f"Preprocessing done, time: {duration_node_preprocessing/60} minutes, {duration_node_preprocessing/3600} hours\n\n"
        )

    # Create the pipeline grid
    grid_pipeline = []
    for pipeline in pipelines_dict_lists:
        pipelines_dict_lists[pipeline]["file_preprocessing_path"] = path_file_list
        print(pipelines_dict_lists[pipeline])
        grid_pipeline_ = ParameterGrid(pipelines_dict_lists[pipeline])

        # Exclude some parameters
        if (
            pipelines_dict_lists_exclude is not None
            and pipeline in pipelines_dict_lists_exclude
        ):
            grid_pipeline_exclude["file_preprocessing_path"] = path_file_list
            grid_pipeline_exclude = ParameterGrid(
                pipelines_dict_lists_exclude[pipeline]
            )
            grid_pipeline_ = [
                params
                for params in grid_pipeline
                if params not in grid_pipeline_exclude
            ]

        grid_pipeline.extend(grid_pipeline_)

    if pipelines_exclude_rules is not None:
        for rule in pipelines_exclude_rules:
            grid_pipeline = [params for params in grid_pipeline if not rule(params)]

    # If we shared the preprocessing across the nodes, we do the all pipeline grid search on every node
    # Else if we performed the same preprocessing on every node, we split the grid search on every node
    if (
        len(grid_preprocessing) < total_nodes
    ):  # We didn't shared the preprocessing across the nodes so we share the pipeline grid search
        grid_pipeline = grid_pipeline[node_index - 1 :: total_nodes]

    # Save some info about the pipeline
    with open(node_log_path, "a") as file:
        file.write(f"### Pipeline ############################\n")
        file.write(f"Number of pipelines: {len(grid_pipeline)}\n")
        file.write(f"Number of pipelines per core: {len(grid_pipeline)//total_cores}\n")
        file.write(
            f"Estimated time for the pipeline: {(len(grid_pipeline)//total_cores) * estimation_time_it/60} minutes, {(len(grid_pipeline)//total_cores) * estimation_time_it/3600} hours\n\n"
        )

        file.write(f"### Grid pipeline ############################\n")
        file.write(f"pipelines: {pipelines_dict_lists}\n\n")

    # Run the training and testing for the current node
    time_start_node_training_testing = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_test_one_core)(settings, pipeline, node_index)
        for pipeline in grid_pipeline
    )
    duration_node_training_testing = time.time() - time_start_node_training_testing

    # Deal with the None values
    results = [result for result in results if result is not None]

    # Save the time of the training and testing
    with open(node_log_path, "a") as file:
        file.write(f"### Results training ############################\n")
        file.write(
            f"Training and testing done, time: {duration_node_training_testing/60} minutes, {duration_node_training_testing/3600} hours\n\n"
        )

    # Save results of this node in a csv file
    results = pd.DataFrame(results)

    # Create the folder if it does not exist
    path_results = settings["PATH_RESULTS"] + f'/V{settings["VERSION"]}/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    # Save the results
    results.to_csv(
        path_results
        + settings["NAME_RESULTS"]
        + f'_V{settings["VERSION"]}'
        + f"_node_{node_index}.csv",
        index=False,
    )

    # Remove the iterations of this node if needed
    if settings["REMOVE_ITERATIONS"]:
        path_results_iterations = (
            settings["PATH_RESULTS"]
            + f'/V{settings["VERSION"]}'
            + settings["FOLDER_RESULTS_ITERATIONS"]
        )
        if os.path.exists(path_results_iterations):
            for file in os.listdir(path_results_iterations):
                if f"_node_{node_index}" in file:
                    os.remove(path_results_iterations + file)

    # Remove the data preprocessed
    if settings["REMOVE_DATA_PROCESSED"]:
        path_data_processed_node = (
            settings["PATH_DATA_PROCESSED"] + "/" + "NODE_" + str(node_index) + "/"
        )
        if os.path.exists(path_data_processed_node):
            for file in os.listdir(path_data_processed_node):
                os.remove(path_data_processed_node + file)

    return results
