from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np


class Input(BaseModel):
    humidity: float


app = FastAPI()
model_bundle = None

# Allow browser access from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from ./static at the app root
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
def load_model():
    global model_bundle
    try:
        model_bundle = joblib.load('model.joblib')
        print('Loaded model.joblib')
    except Exception:
        model_bundle = None
        print('No model.joblib found on startup')


@app.post('/predict')
def predict(item: Input):
    if not model_bundle:
        return {'error': 'model not loaded'}
    poly = model_bundle['poly']
    model = model_bundle['model']
    X = [[item.humidity]]
    X_poly = poly.transform(X)
    y = model.predict(X_poly)
    return {'humidity': item.humidity, 'predicted_temperature': float(y[0])}
