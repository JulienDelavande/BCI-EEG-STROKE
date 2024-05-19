params_dict_lists = { #V2
    'EPOCHS_TMIN': [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5], # 7
    'EPOCHS_LENGTH': [1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5], # 11
    'EPOCHS_EMPTY_FROM_MVT_TMIN': [-4, -4.5, -5, -5.5, -6, -6.5, -7], # 7
    'FMIN': [1], # 1
    'FMAX': [40], # 1
} # 539 iterations, time estimation: ((536//(8*6)) * 11)/60 = 2h (8 nodes, 6 cores each, estimation 11 minutes per iteration)
# DONE

params_dict_lists_exclude = None
params_exclude_rules = None
