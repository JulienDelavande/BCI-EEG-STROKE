# This file contains the settings for the grid search
# Keep the same keys and names for the dictionary

settings = {
    'FILE_PATH': './../../../data/raw/Data_npy/', # Path to the data folder from the script_grid_search.py
    'NUMBER_OF_SESSIONS': -1, # Number of sessions (data files) to be used (-1 for all)
    'RANDOM_STATE': 42, # Random state for the training and testing split
    'N_SPLIT': 4, # Number of splits for the cross-validation
    'BINARY_CLASSIFICATION': True, # Binary classification if true, ternary if false (extension, flexion and no movement)

    "VERSION": 1, # Version of the grid search
    'PATH_RESULTS': './results/', # Path to the results folder from the script_grid_search.py
    'NAME_RESULTS': 'RMLP_MVT_INT_search_preproc', # Name of the results file (version will be concatenated at the end)
    'PATH_DATA_PROCESSED': './data/processed/', # Path to the processed data folder from the script_grid_search.py
    'FOLDER_RESULTS_ITERATIONS': '/iterations/', # Folder to save the results of each iteration inside {PATH_RESULTS}/V{version}/
    'NAME_RESULTS_ITERATIONS': 'RMLP_MVT_INT_search_preproc_iter', # Name of the results file of each iteration (version will be concatenated at the end)
    "LOG_FOLDER": './logs/', # Folder to save the logs from the script_grid_search.py

    "SAVE_EACH_ITERATION": True, # Save the results of each iteration (on parameters set of the grid search in the FOLDER_RESULTS_ITERATIONS folder)
    "REMOVE_ITERATIONS": True, # Remove the results of each iteration after saving them (concatenated in a csv file for each node)
    "REMOVE_DATA_PROCESSED": True, # Remove the processed data after the grid search
}
