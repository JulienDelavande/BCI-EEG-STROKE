# This file contains the Pydantic schemas used for the API
# The schemas are used to validate the data received and sent by the API
# The schemas are also used to generate the documentation of the API

from pydantic import BaseModel
from typing import List, Optional

# Info de tous les signaux d'un enregistrement (un bras)
class SignalInfo(BaseModel):
    """
    Contain all the information about the signals of a recording (one arm)
    """
    duration: float # durées des signaux
    first_movement_time: float # indice temporel du premier mouvement
    last_movement_time: float # indice temporel du dernier mouvement
    electrodes: List[str] # Liste des électrodes utilisées pour l'acquisition des signaux eeg
    electrodes_right_image: Optional[str] = None # Image contenant les plots des électrodes de la partie droite du cerveau (image encodée en base64)
    electrodes_left_image: Optional[str] = None # Image contenant les plots des électrodes de la partie gauche du cerveau (image encodée en base64)

# Info de la session d'un patient (un fichier, qui contient deux enregistrements en général, un par bras) 
class SessionInfo(BaseModel): # -> Envoyé au frontend lors de /upload/
    """
    Contains all the information about a patient session
    """
    patient_id: str # Identifiant du patient
    session_id: str # Identifiant de la session (un patient peu faire plusieurs session, dans une session il fait en général deux enregistrements, un pour le bras gauche et un pour le bras droit, dans chque enregistrement, le patient bouge 10 fois le-dit bras environ toutes les 10 secondes)
    stroke_side: str # 'D' ou 'G' Côté de la lésion cérébrale
    signal_left_arm: Optional[SignalInfo] = None # Informations sur l'enregistrement du bras gauche sur la session (note:L'enregistrement sur le bras gauche peut être absent)
    signal_right_arm: Optional[SignalInfo] = None # Informations sur l'enregistrement du bras droit sur la session (note:L'enregistrement sur le bras droit peut être absent)
    models_available: List[str] # Liste les modèles disponnibles, le frontend devra renvoyer le nom modèle à utiliser
    models_info: dict # Dictionnaire contenant les informations sur les modèles disponnibles (hyperparamètres, etc.)
    data_location: str # 'temp/Data_001_Trial1.npy' le chemin et nom du fichier qui stocke les données brutes du patient, cette info est donnée lors de l'appel à /upload/, le frontend doit renvoyer cette information à l'appel suivant qui est /predict/

# Info de la sélection de l'utilisateur pour la prédiction
class UserSelection(BaseModel): # -> reçu du frontend lors de /predict/
    """
    Contains all the information about the user selection for the prediction
    """
    model: str # nom du modèle choisit pour la prédiction (pour l'instant nom du fichier contenant le modèle sans l'extension)
    bad_channels: Optional[List[str]] = None # [C4, Cz...] La sélections d'électrodes eeg à enlever si None, aucune électrode n'est enlevée
    arm_side: str # (D ou G) Session correspondant au bras de la session à classifer (D ou G)
    time_range: tuple[float, Optional[float]] = [0, None]  # (start_time, end_time) (en s) Début et fin de la session si l'on souhaite couper le signal
    brain_side: str  # (D ou R) Côté du cerveau dont l'on souhaite sélectionner les électrodes, les électrodes sur la tranche centrale du crane sont incluses (Xz)
    data_location: str # 'temp/Data_001_Trial1.npy' le chemin et nom du fichier qui stocke les données brutes du patient, cette info est donnée lors de l'appel à /upload/, le frontend doit renvoyer cette information à l'appel suivant qui est /predict/

# Données utilisée prétraitement des données avant la prédiction
class ProcessingInfo(BaseModel):
    """
    Contains all the information about the data used for the prediction
    """
    patient_id: str # Identifiant du patient
    session_id: str # Identifiant de la session
    stroke_side: str # 'D' ou 'G' Côté de la lésion cérébrale
    arm_side: str # 'D' ou 'G' enregitrement (bras droit ou bras gauche) choisit pour la prédiction
    electrodes: List[str] # Electrodes utilisées pour la prédiction
    time_range: tuple[float, float] # Date de début et date de fin des signaux eeg utilisés pour la prédiction (crop)
    model: str # nom du modèle utilisé pour la prédiction
    data_location: str # 'temp/Data_001_Trial1.npy' le chemin et nom du fichier qui stocke les données brutes du patient
    pipeline_parameters: dict # Paramètres de la pipeline pour la prédiction

# Signaux de l'enregistrement préprocessés et nécéssaires pour les plots côté front-end
class Signals(BaseModel):
    """
    Contains all the signals of the recording preprocessed and necessary for the plots on the front-end
    """
    time: List[float] # Temps
    sfreq: int # Fréquence d'échantillonnage des signaux eeg (et cinématiques après Dataloader) (en Hz) [Note: le Dataloader réenchantillone les signaux cinématiques pour qu'ils aient la même fréquence que les signaux eeg]
    movement_density: Optional[List[float]] = None # Densité de prédiction (mouvement ou intention de mouvement) en fonction du temps
    speed: Optional[List[float]] = None # Signal cinématique de vitesse (en mm/s)
    acceleration: Optional[List[float]] = None # Signal cinématique d'accélération (en mm/s^2)
    movement: Optional[List[float]] = None # Signal cinématique de début de mouvement (début extension = 1 et début flexion = 1, 0 sinon)
    extension: Optional[List[float]] = None # Signal cinématique d'extension du bras (1 si extension, 0 sinon)
    electrodes: Optional[dict] = None # Dictionnaire contenant le nom de l'électrode et son signal (amplitude en uV)

# Résultat de la prédiction
class PredictionResult(BaseModel): # -> Envoyé au frontend lors de /predict/
    """
    Contains all the information about the prediction result
    """
    prediction_density_plot: Optional[str] = None # Pour stocker l'image encodée en base64 du plot de la densité de prédiction (mouvement ou intention de mouvement) en fonction du temps
    signals: Signals # Les signaux de l'enregistrement préprocessés et nécéssaires pour les plots côté front-end
    processing_info: ProcessingInfo # Les informations de prétraitement
    pipeline: dict # Les informations de la pipeline de création du modèle (hyperparamètres preproc, modèle, etc.)