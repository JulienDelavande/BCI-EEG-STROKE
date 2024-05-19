# PIE BCI-EEG-STROKE

## Introduction

The "BCI-EEG-STROKE" project represents a initiative in the realm of stroke rehabilitation, aimed at leveraging brain-computer interface (BCI) technology to facilitate recovery in post-stroke patients, particularly those suffering from hemiplegia—a condition marked by partial paralysis affecting one side of the body. This project is a continuation of the work initiated by the Toulouse University Hospital (CHU), focusing on enhancing patient rehabilitation through innovative neurofeedback applications.

![Illustration of a man in reeducation](./data/static/Readme_illustration.png)

## Background

Stroke is the leading cause of acquired disability in adults, with hemiplegia being one of the most severe consequences, significantly impairing mobility. The cornerstone of this project lies in the concept of neurofeedback, which harnesses the brain's plasticity by encouraging patients to generate movement intentions. These intentions are detected by an algorithm and used to trigger a physical response through an actuator, such as an exoskeleton, thereby promoting cerebral reactivation and rehabilitation.

## Project Goals

Our primary objective is to refine a movement intention detection algorithm, aiming to significantly improve the precision of existing models. By accurately identifying the movement intentions from EEG data during elbow extension and flexion exercises, we aspire to enhance the rehabilitation process for post-stroke patients. The project emphasizes the development of an algorithm capable of discerning specific movement-related patterns in EEG signals, particularly focusing on elbow extension movements on the paralyzed side of the brain to stimulate brain plasticity through neurofeedback.

In addition to algorithmic advancements, user-friendliness and integration ease are prioritized to ensure seamless adoption in medical settings. We aim to deliver clear, optimized code and, if necessary, adapt the previous team's human-machine interface to accommodate our algorithm enhancements, thereby facilitating healthcare professionals in their daily practice.

While the previous team's solution was effective, our project zeroes in on boosting the movement intention detection algorithm's performance. Our efforts are directed towards enhancing accuracy and reliability, ultimately contributing to a more robust and efficient solution that promises to significantly advance post-stroke rehabilitation efforts.

## Achievements and Future Prospects

- **Movement Detection Algorithm**: We have developed a novel algorithm that leverages machine learning techniques to detect movement from EEG signals. Our algorithm has demonstrated promising results, with an accuracy of 86% in intra-session classification, 85% in inter-session classification and 76% in inter-subject classification.

- **Movement Intention Detection**: We have have successfully implemented a movement intention detection algorithm that can accurately identify the intention to perform elbow extension and flexion movements from EEG signals. Our algorithm is still to be tested.

- **User Interface**: We have developed a user-friendly interface that allows healthcare professionals to easily interact with the system. The interface provides real-time feedback on the patient's movement intentions and can be easily integrated into existing rehabilitation protocols.

- **Research and Development**: We have conducted extensive research to identify the most effective machine learning techniques for movement detection and intention classification. Our findings have been instrumental in developing a robust and reliable algorithm.

- **Training and library**: We have developed a training environnement framework and a library to standardize the process of extracting data, training and evaluating the algorithms.

- **Future Prospects**: Our algorithm has the potential to significantly improve the rehabilitation process for post-stroke patients. We are currently exploring opportunities to collaborate with healthcare professionals and rehabilitation centers to further validate and refine our algorithm.

## Usage

To use the application, you can run the following command in the terminal (make sure to have [Make](https://www.gnu.org/software/make/) installed on your machine).

```bash
make setup-app # to setup the application (install dependencies)
make run-app # to run the application (start the backend and the frontend)
```

## Installation for development

Please make sure to have [Python 3.12](https://www.python.org/downloads/release/python-3120/) or higher installed on your machine.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
python -m venv venv_PIE # to create a virtual environment for the project
pip install -r requirements.txt # to install the dependencies
```

## Structure

The project is structured as follows:

- `app/frontend`: contains the source code for the user interface (streamlit)
- `app/backend`: contains the source code for the backend of the application (FastAPI)
- `data/`: contains the data used for training and testing the algorithm
- `docs/`: contains articles, papers and documentation
- `notebooks/`: contains the Jupyter notebooks used for data analysis and algorithm development
- `src/ml_eeg_tools/`: contains the source code for the library used to extract data, train and evaluate the algorithms
- `src/training/`: contains the source code for the training environment
- `gestion_projet/`: contains the project management documentation
- `notes/`: contains the notes taken during the project
- `README.md`: contains the project documentation
- `requirements.txt`: contains the list of dependencies
- `Makefile`: contains the commands to setup and run the application
- `setup.py`: contains the package information
- `run_app.bat`: contains the command to run the application on Windows
- `setup_app.bat`: contains the command to setup the application on Windows
- `.gitignore`: contains the list of files to ignore

Most folders contains a `README.md` file with more detailed information about the content.

## Contributers (PIE - 2023-2024)

- **Project Manager (Sisy, Neuro-IA)**: Jules Gomel
- **Data Scientist (SDD, Neuro-IA)**: Mathieu Dario
- **Data Scientist (SDD, Neuro-IA)**: Rayanne Igbida
- **Data Scientist (OTSU, Neuro-IA)**: Aurélien Deniau
- **Data Scientist (SD-IF, Neuro-IA)**: Brice Appenzeller
- **Data Scientist (SDD, Neuro-IA)**: Julien Delavande
