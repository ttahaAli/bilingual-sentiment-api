from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
import re

# Initialize the FastAPI Application
app = FastAPI(
    title="Bilingual Sentiment Analysis API",
    description="FastAPI endpoint deploying an ANN to classify Roman Urdu & English reviews.",
    version="1.0"
)

# Define the ANN Architecture (Must match train_ann.py exactly!)
class SentimentANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Define the Pydantic validation schemas
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    original_text: str
    cleaned_text: str
    prediction_id: int
    sentiment: str
    confidence_scores: dict

# ==========================================
# MODEL & VECTORIZER INITIALIZATION (ON RUNTIME)
# ==========================================
# Modified to match your actual model checkpoint footprint (64 inputs instead of 5000)
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 3

# Absolute paths formatted as Windows raw strings to prevent escaping issues
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/ann_baseline.pt"
# 1. Load fitted TfidfVectorizer
try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Loaded TfidfVectorizer successfully.")
except FileNotFoundError:
    raise RuntimeError(f"🚨 Vectorizer file not found at: {VECTORIZER_PATH}")

# 2. Initialize and load PyTorch model weights
try:
    model = SentimentANN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    # Using weights_only=True to silence the PyTorch security warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()  # Disables dropout layers for stable inference
    print("✅ Loaded PyTorch SentimentANN weights successfully.")
except FileNotFoundError:
    raise RuntimeError(f"🚨 Model checkpoint not found at: {MODEL_PATH}")

# 3. Text cleaning function
def clean_roman_urdu(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# API ROUTING ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "Bilingual Sentiment ANN",
        "features_expected": INPUT_DIM
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input review text cannot be empty.")
    
    cleaned_text = clean_roman_urdu(request.text)
    
    if not cleaned_text:
        cleaned_text = "neutral" 
    
    # 1. Transform text using loaded vectorizer
    vectorized = vectorizer.transform([cleaned_text]).toarray()
    input_tensor = torch.tensor(vectorized, dtype=torch.float32)
    
    # 2. Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        prediction_idx = int(torch.argmax(outputs, dim=1).item())
        
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_sentiment = sentiment_map[prediction_idx]
    
    return PredictionResponse(
        original_text=request.text,
        cleaned_text=cleaned_text,
        prediction_id=prediction_idx,
        sentiment=predicted_sentiment,
        confidence_scores={
            "Negative": float(probabilities[0]),
            "Neutral": float(probabilities[1]),
            "Positive": float(probabilities[2])
        }
    )