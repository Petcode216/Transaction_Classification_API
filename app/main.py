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
MODEL_NAME = "transaction_classifier.pkl"
MODEL_PATH = MODEL_NAME
print(MODEL_PATH)
classifier = TransactionClassifier()

try:
    classifier.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
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
