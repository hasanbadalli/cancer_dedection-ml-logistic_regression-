from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "model.pkl")
with open(BASE_DIR / "models" / "metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold_benign"]
LABELS = metadata["labels"]

app = FastAPI()

class PatientInput(BaseModel):
    features: list[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(patient: PatientInput):
    proba = model.predict_proba([patient.features])[0]

    p_malignant = float(proba[0])
    p_benign = float(proba[1])

    prediction = 1 if p_benign > THRESHOLD else 0

    return {
        "prediction": LABELS[str(prediction)],
        "threshold_used": THRESHOLD,
        "probability": {
            "malignant": p_malignant,
            "benign": p_benign
        }
    }

