# This file contains the parameters for the grid search of the preprocessing
# Keep the same keys and names for the dictionaries and lists

# params_dict_lists : paramteres to be used in the grid search for the preprocessing
params_dict_lists = { #V1
    'EPOCHS_TMIN': [-1, -0.5, 0], # 3 -> start time of the epoch from the movement onset (in seconds)
    'EPOCHS_LENGTH': [4, 3, 2.5, 2, 1.8, 1.7, 1.5, 1.2, 1, 0.5], # 10 -> length of the epoch (in seconds) [TO BE USED IF NO EPOCHS_TMAX IS USED]
    # 'EPOCHS_TMAX': [0, 0.5, 1], # 3 -> end time of the epoch from the movement onset (in seconds) [TO BE USED IF NO EPOCHS_LENGTH IS USED]
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [-8, -7, -6, -5, -4], # 5 -> start time of the empty epoch from the movement onset
    'EPOCHS_INTENTION_FROM_MVT_TMIN': [-4, -3, -2.8, -2.5, -2.2, -2, -1.8, -1.5, -1.2], # 9 -> OPTIONAL: start time of the intention epoch from the movement onset
    'FMIN': [9], # 1 -> min frequency
    'FMAX': [30], # 1 -> max frequency
} # 3*10*5*9*1*1 = 1350 combinations -> 8 nodes, 6cores/node 1350/48 = 28.125 -> * 11min = 5h

# params_dict_lists_exclude : parameters to be excluded from the grid search for the preprocessing (same keys as params_dict_lists)
params_dict_lists_exclude = None

# params_exclude_rules : rules to exclude some parameters combinations
# Example: params_exclude_rules = [lambda params: params['FMIN'] >= params['FMAX'], ...]
params_exclude_rules = [lambda params: params['EPOCHS_INTENTION_FROM_MVT_TMIN'] + params['EPOCHS_LENGTH'] > 0]
