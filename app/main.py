import os
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.classifier import TransactionClassifier

app = FastAPI(
    title="Transaction Classifier API",
    description="API for classifying financial transactions",
    version="1.0.0"
)

# 1. Determine absolute MODEL_PATH
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root (/app)
MODEL_NAME = "transaction_classifier.pkl"
default_model_path = BASE_DIR / "models" / MODEL_NAME
MODEL_PATH = Path(os.getenv("MODEL_PATH", default_model_path))

print(f"Looking for model at: {MODEL_PATH}")

# 2. Instantiate and load classifier
classifier = TransactionClassifier()
try:
    classifier.load(str(MODEL_PATH))
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    classifier = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

# 3. Request/Response schemas
class TransactionRequest(BaseModel):
    text: str

class TransactionResponse(BaseModel):
    category: str
    amount: int
    confidence: float

# 4. Endpoints
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "model_loaded": classifier is not None
    }

@app.post("/predict", response_model=TransactionResponse)
async def predict_transaction(request: TransactionRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    try:
        category, amount, confidence = classifier.predict(request.text)
        return TransactionResponse(
            category=category,
            amount=amount,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
