# This file contains the parameters for the grid search of the preprocessing
# Keep the same keys and names for the dictionaries and lists

# params_dict_lists : paramteres to be used in the grid search for the preprocessing
params_dict_lists = { #V1
    'EPOCHS_TMIN': [-1, -0.5, 0], # 3 -> start time of the epoch from the movement onset (in seconds)
    'EPOCHS_LENGTH': [-5, -4.5, -4], # 3 -> length of the epoch (in seconds) [TO BE USED IF NO EPOCHS_TMAX IS USED]
    # 'EPOCHS_TMAX': [0, 0.5, 1], # 3 -> end time of the epoch from the movement onset (in seconds) [TO BE USED IF NO EPOCHS_LENGTH IS USED]
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [2.5, 3, 3.5], # 3 -> start time of the empty epoch from the movement onset
    'EPOCHS_INTENTION_FROM_MVT_TMIN': [5, 6], # 2 -> OPTIONAL: start time of the intention epoch from the movement onset
    'FMIN': [1], # 1 -> min frequency
    'FMAX': [40], # 1 -> max frequency
} # 3*3*3*2*1*1 = 54 combinations -> 8 nodes, 6cores/node

# params_dict_lists_exclude : parameters to be excluded from the grid search for the preprocessing (same keys as params_dict_lists)
params_dict_lists_exclude = None

# params_exclude_rules : rules to exclude some parameters combinations
# Example: params_exclude_rules = [lambda params: params['FMIN'] >= params['FMAX'], ...]
params_exclude_rules = None
