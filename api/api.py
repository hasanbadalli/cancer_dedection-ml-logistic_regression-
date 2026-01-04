from fastapi import FastAPI
import joblib, json
from pathlib import Path
from api.schemas import PatientInput
from api.features import FEATURE_ORDER
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "model.pkl")
with open(BASE_DIR / "models" / "metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold_benign"]
LABELS = metadata["labels"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(patient: PatientInput):

    feature_dict = patient.model_dump()
    features = [feature_dict[name] for name in FEATURE_ORDER]

    proba = model.predict_proba([features])[0]

    p_malignant = float(proba[0])
    p_benign = float(proba[1])

    prediction = 1 if p_benign >= THRESHOLD else 0

    return {
        "prediction": LABELS[str(prediction)],
        "threshold_used": THRESHOLD,
        "probability": {
            "malignant": p_malignant,
            "benign": p_benign
        }
    }
