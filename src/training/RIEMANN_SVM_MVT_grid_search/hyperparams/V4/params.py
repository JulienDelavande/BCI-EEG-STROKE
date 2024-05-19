params_dict_lists = { #V4
    'EPOCHS_TMIN': [-0.75], # 1
    'EPOCHS_LENGTH': [1.7], # 1
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [-6], # 1
    'FMIN': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
             24, 25, 26, 27, 28, 29, 30, 35, 40], # 32
    'FMAX': [10, 15, 20, 25, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 60, 65, 70, 75], # 19
} # 32*19 = 608 combinations
## PENDING

params_dict_lists_exclude = None

params_exclude_rules = [
    lambda params: params['FMIN'] >= params['FMAX'],
]