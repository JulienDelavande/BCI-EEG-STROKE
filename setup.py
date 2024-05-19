from setuptools import setup, find_packages

setup(
    name="ml_eeg_tools",  # Le nom de votre paquet
    version="0.1.0",
    author="Votre Nom",
    description="Un paquet pour le prétraitement, l'entraînement et le test de modèles de machine learning sur des données EEG.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "mne",
        "pyriemann",
        "scipy",
        "matplotlib",
    ],
)
