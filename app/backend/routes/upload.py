from fastapi import APIRouter, File, UploadFile, HTTPException
from services import data_extraction
from schemas import schemas

router = APIRouter()
TEMP_FOLDER = "temp"
DATA_NAME = "Data"


@router.post("/", response_model=schemas.SessionInfo)
async def upload_file(file: UploadFile = File(...)):
    """
    Receives the file and returns the session information.

    Parameters:
    ----------
        file (UploadFile): The file to be received.

    Returns:
    --------
        schemas.SessionInfo: The session information.
    """
    try:
        # Supprimer les fichiers temporaires
        await data_extraction.remove_temporary_files()

        # Enregistrer temporairement le fichier reçu
        data_location = await data_extraction.save_file_temporarily(file)

        # Convertir en .npy si le fichier est .mat
        if data_location.endswith(".mat"):
            data_location = data_extraction.mat_to_npy(
                data_location, TEMP_FOLDER, DATA_NAME
            )

        # Extraire les informations de session du fichier
        session_info = data_extraction.extract_data_session(data_location)

        # Retourner les informations au frontend pour affichage
        return session_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
