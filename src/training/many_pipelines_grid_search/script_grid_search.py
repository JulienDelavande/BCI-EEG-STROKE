from ml_eeg_tools.train.grid_search import run_search_one_node, parse_args
import importlib

ESTIMATION_TIME_IT = 660  # seconds

args = parse_args()
node_index = args.node_index
total_nodes = args.total_nodes
version = args.version
cores_per_node = args.cores_per_node

# Import the settings, params_list and pipeline
settings = importlib.import_module(f"hyperparams.V{version}.settings").settings
params_dict_lists = importlib.import_module(
    f"hyperparams.V{version}.params"
).params_dict_lists
pipelines_dict_lists = importlib.import_module(
    f"hyperparams.V{version}.pipeline"
).pipelines_dict_lists
params_dict_lists_exclude = (
    importlib.import_module(f"hyperparams.V{version}.params").params_dict_lists_exclude
    if hasattr(
        importlib.import_module(f"hyperparams.V{version}.params"),
        "params_dict_lists_exclude",
    )
    else None
)
pipelines_dict_lists_exclude = (
    importlib.import_module(
        f"hyperparams.V{version}.pipeline"
    ).pipelines_dict_lists_exclude
    if hasattr(
        importlib.import_module(f"hyperparams.V{version}.pipeline"),
        "pipelines_dict_lists_exclude",
    )
    else None
)
params_exclude_rules = (
    importlib.import_module(f"hyperparams.V{version}.params").params_exclude_rules
    if hasattr(
        importlib.import_module(f"hyperparams.V{version}.params"),
        "params_exclude_rules",
    )
    else None
)
pipelines_exclude_rules = (
    importlib.import_module(f"hyperparams.V{version}.pipeline").pipelines_exclude_rules
    if hasattr(
        importlib.import_module(f"hyperparams.V{version}.pipeline"),
        "pipelines_exclude_rules",
    )
    else None
)

settings['VERSION'] = version

# Run the search
run_search_one_node(
    settings=settings,
    params_dict_lists=params_dict_lists,
    pipelines_dict_lists=pipelines_dict_lists,
    params_dict_lists_exclude=params_dict_lists_exclude,
    pipelines_dict_lists_exclude=pipelines_dict_lists_exclude,
    params_exclude_rules=params_exclude_rules,
    pipelines_exclude_rules=pipelines_exclude_rules,
    node_index=node_index,
    total_nodes=total_nodes,
    total_cores=cores_per_node,
    estimation_time_it=ESTIMATION_TIME_IT,
)
