from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Health check for the backend.

    Returns:
    --------
        dict: The status of the backend.
    """
    return {"status": "ok", "message": "Backend is up and running!"}
