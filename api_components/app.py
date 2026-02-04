from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from keras.models import load_model
app = FastAPI()

model = load_model("tuned_nn_model.keras")
scaler = joblib.load("scaler.joblib")
features = joblib.load("features.joblib")


class PatientData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    radius_se: float
    perimeter_se: float
    area_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post("/predict")
def predict(data: PatientData):

    df = pd.DataFrame([data.dict()])

    X = df[features]

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    return {"prediction": int(pred)}
