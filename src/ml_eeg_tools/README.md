# ML EEG TOOLS

This is the built-in library for EEG data processing and machine learning in the context of the PIE stroke rehabilitation project. Every time of piece of code is written mutiple time in mutiple use cases (Notebooks, training, backen, front, end, etc.), it should be written here.

## Installation

Please make sure to have Python 3.12 or higher installed on your machine.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.
It's using the requirements.txt file from the whole project.

From the root of the project, run the following command:

```bash
pip install -r requirements.txt
```

## Structure

The library is divided into 3 main parts (for now):

- **preprocessing**: Contains all the functions to preprocess the EEG data.
- **test_model**: Contains all the functions to test the model.
- **train**: Contains all the functions to train the model.

### Preprocessing

The preprocessing part contains all the functions to preprocess the EEG data. It is divided into 3 subparts:

- **data_extraction**: Contains mat_to_npy function to convert the .mat files to .npy files.\
*Functions:*
  - `mat_to_npy`: Convert the .mat files recorded in the CHU to .npy files.
- **data_loader**: Contains the DataLoader class to load the EEG data.
*Classes:*
  - `DataLoader`: Load the EEG data from the .npy files.
- **data_prepation**: Contains all the functions to prepare the EEG data for the model. (preprocessing : filtering, epoching, etc.)
*Functions:*
  - `prepare_data_train`: preprocess the data for training/testing (filtering, epoching, etc.)
  - `get_channels`: get the channels of the stroke side.

### Test Model

The test_model part contains all the functions to test the model. It is divided into 1 subpart:

- **test_epoch**: Contains all the functions to test the model. It test the accuracy on the epoched data through the 4 tests.\
*Functions:*
  - `train_test_A`: train and test with cross validation for data from the same session for training and testing,
  - `train_test_B`: train and test with cross validation for data from the same patient for training and testing but different session,
  - `train_test_C`: train and test with cross validation for data from different patient for training and testing,
  - `train_test_X`: train and test with cross validation for all epoched data from all patients for training and testing shuffled.

### Train

The train part contains all the functions to train the model. It is divided into 1 subpart:

- **grid_search**: Contains all the functions to train the model on mutiple cores and nodes. It first paralellize the preprocessing and save the preprocessed data. Then, it paralellize the grid search on pipeline and save the results.\
*Functions:*
  - `run_search_one_node`: Run the grid search on one node.
  - `prepare_data_one_core`: Preprocess the data on one core.
  - `train_test_one_core`: Train and test the model on one core.
  - `parse_args`: Parse the arguments from the command line.
