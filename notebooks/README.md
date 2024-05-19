# Notebooks folder documentation

This folder contains the Jupyter notebooks used for data analysis and algorithm development. Some of the notebooks are really good insights into the methods we used to understand the data, label it and train the models. Others are just more like a draft of the final code. Notebooks to see to better understand the project are ranked with stars.

The notebooks are organized as follows:

- [`exploration_data/ *`](exploration_data/): contains the notebooks used for data exploration
  - [`exploration_data/exploration_aurelien_visu.ipynb *`](./exploration_data/exploration_aurelien_visu.ipynb): Some visualizations of the data
  - [`exploration_data/exploration_data_structure.ipynb **`](./exploration_data/exploration_data_structure.ipynb): Exploration of the data structure -> .npy files
  - [`exploration_data/labelling_data.ipynb **`](./exploration_data/labelling_data.ipynb): Insights on how re-label the data (for extension and flexion onset)
  - [`exploration_data/movement_timing_analysis.ipynb *`](./exploration_data/movement_timing_analysis.ipynb): Analysis of duration between flexion and extension onset.
  - [`exploration_data/plotting.ipynb`](./exploration_data/plotting.ipynb): Some plotting draft for eeg channels visualization (frontend requirement)
  - [`exploration_data/Visualisation_MNE.ipynb`](./exploration_data/Visualisation_MNE.ipynb): Visualisation of the data with MNE library
- [`exploration_pipelines/ **`](exploration_pipelines/): contains the notebooks used for pipeline exploration
  - [`exploration_pipelines/pipeline_riemann.ipynb **`](./exploration_pipelines/pipeline_riemann.ipynb): Exploration of the Riemannian pipeline
  - [`exploration_pipelines/pipeline_xdawn_csp.ipynb **`](./exploration_pipelines/pipeline_xdawn_csp.ipynb): Exploration of the XDawn and CSP pipeline
  - [`exploration_pipelines/pipeline_BP.ipynb`](./exploration_pipelines/pipeline_BP.ipynb): Exploration of the Band power pipeline
  - [`exploration_pipelines/pipeline_template.ipynb **`](./exploration_pipelines/pipeline_template.ipynb): Template for pipeline exploration. It is also an exploration of a pipeline with a simple standardization and a classifier
  - [`exploration_pipelines/Visualisation CSP + Kmeans + Voting.ipynb`](./exploration_pipelines/Visualisation%20CSP%20%2B%20Kmeans%20%2B%20Voting.ipynb): Visualization of the CSP + Kmeans + Voting pipeline
  - [`\non_standard\`](./exploration_pipelines/non_standard/): contains the notebooks used pipeline exploration before stdandardization
- [`method_golden_standard/`](method_golden_standard/): contains the notebooks we took inspiration from to perform our research
  - [`method_golden_standard/BE_MI_filled.ipynb *`](./method_golden_standard/BE_MI_filled.ipynb): BE of Neuro IA courses for EEG classification filled
- [`exploration_grid_search/`](exploration_grid_search/): contains the notebooks used for grid search exploration
  - [`exploration_grid_search/preprocessing_hyperparameter_search.ipynb`](./exploration_grid_search/preprocessing_hyperparameter_search.ipynb): Exploration of the preprocessing hyperparameter search
- [`analysis/ **`](analysis/): contains the notebooks used for analysis of the grid search results (grid search performed on the ISAE SUPAERO HPC cluster - Rainman in the `src/training/` folder)
  - [`analysis/hyperparam_search_RIEMANN_SVM_MVT.ipynb **`](./analysis/hyperparam_search_RIEMANN_SVM_MVT.ipynb): Analysis of the hyperparameters for the Riemannian pipeline with SVM used for movement detection
  - [`analysis/hyperparam_search_RIEMANN_SVM_MVT_INT.ipynb **`](./analysis/hyperparam_search_RIEMANN_SVM_MVT_INT.ipynb): Analysis of the hyperparameters for the Riemannian pipeline with SVM used for movement intention detection
  - [`analysis/hyperparam_search_many_pipelines_MVT.ipynb **`](./analysis/hyperparam_search_many_pipelines_MVT.ipynb): Grid search on many pipelines and models for movement detection
- [`ml_eeg_tools_example/ *`](ml_eeg_tools_example/): contains the notebooks used to explain and develop the ml_eeg_tools library
  - [`ml_eeg_tools_example/data_preparation.ipynb *`](./ml_eeg_tools_example/data_preparation.ipynb): Example of data preparation (preprocessing) for the model (fonctionalities of the ml_eeg_tools library)

Most of the notebooks are commented so that you can understand the process and the results.
