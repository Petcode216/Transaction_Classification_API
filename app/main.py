from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.classifier import TransactionClassifier 
import os
from pathlib import Path
from fastapi.responses import RedirectResponse


app = FastAPI(
    title="Transaction Classifier API",
    description="API for classifying financial transactions",
    version="1.0.0"
)

# Load model khi khởi động
# Determine base directory (where this file lives)
BASE_DIR = Path(__file__).resolve().parent

# Name of your file
MODEL_NAME = "transaction_classifier.pkl"

# Default to /app/models/..., overrideable by ENV
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR/ MODEL_NAME))

print(f"Looking for model at: {MODEL_PATH}")

# Initialize your classifier
classifier = TransactionClassifier()

# Startup: load the model once
try:
    classifier.load(str(MODEL_PATH))
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    classifier = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None


# Request/Response models
class TransactionRequest(BaseModel):
    text: str

class TransactionResponse(BaseModel):
    category: str
    amount: int
    confidence: float

# Health check endpoint
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "model_loaded": classifier is not None
    }

# Prediction endpoint
@app.post("/predict", response_model=TransactionResponse)
async def predict_transaction(request: TransactionRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        category, amount, confidence = classifier.predict(request.text)
        return {
            "category": category,
            "amount": amount,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Chạy với uvicorn khi thực thi trực tiếp
if __name__ == "__main__":
    import uvicorn
    print("🌐 Server running at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
