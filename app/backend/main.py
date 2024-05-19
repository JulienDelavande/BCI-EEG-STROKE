from fastapi import FastAPI
from routes import health_check as health_check_router
from routes import upload as upload_router
from routes import prediction as prediction_router

app = FastAPI()

app.include_router(health_check_router.router, tags=["Health Check"], prefix="/health")
app.include_router(upload_router.router, tags=["File Upload"], prefix="/upload")
app.include_router(prediction_router.router, tags=["Prediction"], prefix="/predict")
