# Backend of the app of the BCI-EEG-STROKE project

This is the backend of the EEG BCI Stroke App. It is a FastAPi application that provides the API for the frontend to make predictions on eeg data.

## Installation

To install the backend, you need to have Python 3.8 or higher installed. Then you can install the backend by running the following command:

```bash
python -m venv venv_backend
pip install -r requirements.txt
```

or

```bash
make setup-backend
```

## Usage

To start the backend, you can run the following command:

```bash
uvicorn main:app --reload
```

or

```bash
make run-backend
```

This will start the backend on `http://127.0.0.1:8000`. You can then use the frontend to make predictions on eeg data.

## API Documentation

The API documentation can be found at `http://127.0.0.1:8000/docs`. This is a Swagger UI that allows you to test the API endpoints.

Quick summary of the API endpoints:

- `/health`: [GET] Check if the backend is running
- `/upload`: [POST] Upload eeg data from a file
- `/predict`: [POST] Make a prediction on eeg data

## Structure

The backend is structured as follows:

- `main.py`: contains the FastAPI application
- `routes/`: contains the API endpoints
- `schemas/`: contains the Pydantic models used for the API
- `models/`: contains the machine learning models and pipelines settings
- `services/`: contains the services used to prepare the data, make predictions and plot the data
- `temp/`: contains the temporary files used for the API
- `hooks/`: contains the hooks used for building the .exe
- `tests.ipynb/`: contains the tests for the API endpoints (helps to understand how to use the API)
- `requirements.txt`: contains the list of dependencies
- `README.md`: contains the backend documentation
