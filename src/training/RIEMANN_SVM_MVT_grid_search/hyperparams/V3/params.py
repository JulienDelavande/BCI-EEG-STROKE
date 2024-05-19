params_dict_lists = { #V3
    'EPOCHS_TMIN': [-0.75], # 1
    'EPOCHS_LENGTH': [1.7], # 1
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [-6], # 1
    'FMIN': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # 10
    'FMAX': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75], # 10
} # 100 iterations, time estimation: ((100//(8*6)) * 11)/60 = 0.4h (8 nodes, 6 cores each, estimation 11 minutes per iteration)
## DONE

params_dict_lists_exclude = None
params_exclude_rules = None
