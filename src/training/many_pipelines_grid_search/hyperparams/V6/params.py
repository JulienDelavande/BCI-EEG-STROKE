# This file contains the parameters for the grid search of the preprocessing
# Keep the same keys and names for the dictionaries and lists

# params_dict_lists : paramteres to be used in the grid search for the preprocessing
params_dict_lists = { #V1
    'EPOCHS_TMIN': [-1], # 1 -> start time of the epoch from the movement onset (in seconds)
    'EPOCHS_LENGTH': [3], # 5 -> length of the epoch (in seconds) [TO BE USED IF NO EPOCHS_TMAX IS USED]
    # 'EPOCHS_TMAX': [0, 0.5, 1], # 3 -> end time of the epoch from the movement onset (in seconds) [TO BE USED IF NO EPOCHS_LENGTH IS USED]
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [-4], # 1 -> start time of the empty epoch from the movement onset
    #'EPOCHS_INTENTION_FROM_MVT_TMIN': [-4, -1.2, -1, -0.5], # 4 -> OPTIONAL: start time of the intention epoch from the movement onset
    'FMIN': [1], #  7 -> min frequency
    'FMAX': [40], # 7 -> max frequency
} # 1*5*1*4*7*7 = 980 combinations -> 8 nodes, 6cores/node 980/48 = 20.42 -> * 11min = 3h

# params_dict_lists_exclude : parameters to be excluded from the grid search for the preprocessing (same keys as params_dict_lists)
params_dict_lists_exclude = None

# params_exclude_rules : rules to exclude some parameters combinations
# Example: params_exclude_rules = [lambda params: params['FMIN'] >= params['FMAX'], ...]
params_exclude_rules = None

