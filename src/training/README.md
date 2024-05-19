# Training folder

This folder contains the training code and hyperparameters search for the PIE project. It aims to be consider as a training environment to be deployed on the ISAE Supaero's cluster for the computation of the hyperparameters search and model training. It contains templates for the grid search and training on the cluster for scki-learn like pipeline and preprocessing available in the eeg_ml_tools library.

## Already searched pipelines

- **RIEMANN_SVM_MVT**: Riemannian SVM pipeline for movement detection grid search.
- **RIEMANN_SVM_MVT_INT**: Riemannian SVM pipeline for intention of movement detection grid search.
- **many_pipelines_MVT**: Many pipelines grid search for movement detection and mouvement intention.

## Already trained models

- **RIEMANN_SVM_MVT**: Riemannian SVM model for movement detection.
- **RIEMANN_SVM_MVT_INT**: Riemannian SVM model for intention of movement detection.

## Structure

- [**template_grid_search/**](template_grid_search): Contains the code for the grid search of one pipeline.
  - [**script_grid_search.py**](template_grid_search/script_grid_search.py): Contains the python script to perform the grid search on one node. It uses the hyperparameters from the the `hyperparams` folder to train the model and save the results in the `results` folder. It saves intermediates preprocessed data in the `data` folder. It first paralellize the preprocessing and save the preprocessed data. Then, it paralellize the grid search on pipeline and save the results.
  - [**job_template.slurm**](template_grid_search/job_template.slurm): Contains the template for the slurm script to launch the grid search on the cluster. It uses the `script_grid_search.py` to perform the grid search and save the results in the `results` folder.
  - [**many_jobs.sh**](template_grid_search/many_jobs.sh): Contains the script to launch the grid search on the cluster. It uses the `job_template.slurm` to perform the grid search and save the results in the `results` folder, the logs in the `logs`.
  - [**hyperparams/**](template_grid_search/hyperparams): Contains the hyperparameters for the grid search.
    - **V{X}/params.py**: Contains the hyperparameters for the preprocessing for the grid search. X is the version of the hyperparameters search.
    - **V{X}/pipeline.py**: Contains the hyperparameters for the pipeline for the grid search.
    - **V{X}/settings.py**: Contains the parameters for the settings for the grid search.
  - **results/V{x}**: Contains the results of the grid search.
    - **iterations/**: Contains the csv files of the result of one iteration of the grid search. Each file contain one line with the hyperparameters and the score of the model for this set of hyperparameters. This folder can automatically deleted at the end since all results are gathered for each node in the `results/V{x}/{NAME}_V{X}_Y.csv` file. X is the version of the hyperparameters search. Y is the index of the node.
  - [**logs/**](template_grid_search/logs): Contains the logs of the grid search. Slurm logs are saved here.
    - **V{X}/**: Contains the logs of the grid search. X is the version of the hyperparameters search.
  - **data/proprecessed/V{X}/NODE_{Y}**: Contains the preprocessed data of the grid search (intermediates). X is the version of the hyperparameters search. Y is the index of the node. It can be automatically deleted at the end.

- [**train/**](train): Contains the code for training a model and saving it.
  - [**sript_train.py**](train/script_train.py): Contains the python script to train the model and save it. The hyperparameters should be set in the script.
  - [**job.slurm**](train/job.slurm): Contains the template for the slurm script to launch the training on the cluster. It uses the `script_train.py` to train the model and save it.
  - [**logs/**](train/logs): Contains the logs of the training. Slurm logs are saved here.
  - [**models/**](train/models): Contains the trained models.
  - [**results/**](train/results): Contains the results of the training (scores, etc.).

- [**RIEMANN_SVM_MVT_grid_search**](RIEMANN_SVM_MVT_grid_search): Contains the code for the grid search of the Riemannian SVM MVT model. (based on the template_grid_search)

- [**RIEMANN_SVM_MVT_INT_grid_search**](RIEMANN_SVM_MVT_INT_grid_search): Contains the code for the grid search of the Riemannian SVM INT MVT model. (based on the template_grid_search)

- [**RIEMANN_SVM_MVT_train**](RIEMANN_SVM_MVT_train): Contains the code for the training of the Riemannian SVM MVT model. (based on the train)

- [**RIEMANN_SVM_MVT_INT_train**](RIEMANN_SVM_MVT_INT_train): Contains the code for the training of the Riemannian SVM INT MVT model. (based on the train)

- [**many_pipelines_grid_search/**](many_pipelines_grid_search): Contains the code for the grid search of many pipelines. (based on the template_grid_search)

## Installation

Please make sure to have Python 3.8.8 as it is the version used on the cluster.
Connect to the cluster and clone the repository.
To connect to the cluster, ask for the access to Rainman and connect through the online interface: [Rainman](https://icare.isae.fr/)

```bash
git clone https://github.com/mdario-github/PIE_2023.git
```

Install the requirements.

```bash
cd PIE_2023/src/training
pip install -r requirements.txt
```

## Usage

Lunch the grid search on the cluster. {MODEL} is the model to train (RIEMANN_SVM_MVT or RIEMANN_SVM_MVT_INT for instance). {VERSION} is the version of the hyperparameters search (1, 2, 3 ... for instance).

```bash
cd PIE_2023/src/training/{MODEL}_grid_search
chmod +x many_jobs.sh # Make the script executable if it is not already
bash many_jobs.sh {VERSION}
```

To lunch the training on the cluster. {MODEL} is the model to train (RIEMANN_SVM_MVT or RIEMANN_SVM_MVT_INT for instance).

```bash
cd PIE_2023/src/training/train
sbatch job.slurm
```

Some useful commands to monitor the jobs on the cluster.

```bash
squeue -u <USER> # To see the jobs of a user
scontrol show job <JOB_ID> # To see the details of a job
scancel <JOB_ID> # To cancel a job
sacct -u <USER> # To see the history of the jobs of a user
sinfo # To see the nodes of the cluster
```
