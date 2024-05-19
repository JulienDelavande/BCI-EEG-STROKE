from fastapi import APIRouter, HTTPException
from schemas import schemas
from services import (
    prediction_service,
    preprocessing_service,
    plot_service,
    data_extraction,
)
from models.pipelines import pipelines

router = APIRouter()

TEMP_FOLDER = "temp"
PLOT_NAME = "density_plot.png"
THRESHOLD_DENSITY_PREDICTION = 0.5
PLOT = True
PLOT_DENSITY = True


@router.post("", response_model=schemas.PredictionResult)
async def predict(user_selection: schemas.UserSelection):
    """
    Receives the user selection and returns the prediction result.

    Parameters:
    ----------
        user_selection (schemas.UserSelection): The user selection.

    Returns:
    --------
        schemas.PredictionResult: The prediction result.
    """
    try:
        pipeline = pipelines[user_selection.model]

        # Prétraitement basé sur la sélection de l'utilisateur
        raws, processing_info = preprocessing_service.prepare_data(
            user_selection, pipeline
        )

        # Transformation des données eeg en fenêtres glissantes pour traitement dans le classifieur
        windows = preprocessing_service.creating_sliding_windows(raws, pipeline)

        # Prédiction basés sur les données EEG reçues
        predictions, pipeline_info = prediction_service.predict(
            windows, processing_info
        )

        # Reconstruction de la prédiction sur le signal temporel (densité de prédiction)
        movement_density = prediction_service.density_on_prediction(
            raws, predictions, pipeline
        )
        movement_times = plot_service.density_plot(
            raws,
            movement_density,
            threshold=THRESHOLD_DENSITY_PREDICTION,
            plot=PLOT,
            plot_density=PLOT_DENSITY,
            name=f"{TEMP_FOLDER}/{PLOT_NAME}",
        )
        encoded_string = (
            plot_service.encode_image_to_base64(f"{TEMP_FOLDER}/{PLOT_NAME}")
            if PLOT
            else None
        )

        signals = schemas.Signals(
            time=list(raws.times),
            movement_density=list(movement_density),
            sfreq=raws.info["sfreq"],
            speed=list(raws["VAC"][0][0]),
        )
        prediction_result = schemas.PredictionResult(
            prediction_density_plot=encoded_string if PLOT else None,
            signals=signals,
            processing_info=processing_info,
            pipeline=pipeline,
        )

        # Supprimer les fichiers temporaires
        # await data_extraction.remove_temporary_files()

        # Retourner le résultat sous forme d'un objet PredictionResult
        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
