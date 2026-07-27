# 🌍 Bilingual Sentiment Analysis API (Roman Urdu & English)

A production-ready REST API built with **FastAPI**, **PyTorch**, and **Docker** to classify sentiment in code-switched Roman Urdu and English reviews.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **ML Framework:** PyTorch (Artificial Neural Network)
* **Feature Extraction:** TF-IDF Vectorizer (scikit-learn)
* **API Framework:** FastAPI & Uvicorn
* **Containerization:** Docker

---

## ⚡ Quickstart with Docker

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/bilingual-sentiment-api.git](https://github.com/your-username/bilingual-sentiment-api.git)
   cd bilingual-sentiment-api

## Build the Docker Image: 

## Bash
docker build -t bilingual-sentiment-api .
Run the container:

## Bash
docker run -d -p 8000:8000 --name sentiment-app bilingual-sentiment-api
Access the interactive API docs:
Open http://localhost:8000/docs in your browser.

## 🧪 Example API Request & Response
Request
Bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "bohat achi service hai, mazeedar food"}'

## Response
JSON
{
  "original_text": "bohat achi service hai, mazeedar food",
  "cleaned_text": "bohat achi service hai mazeedar food",
  "prediction_id": 2,
  "sentiment": "Positive",
  "confidence_scores": {
    "Negative": 0.0001,
    "Neutral": 0.0012,
    "Positive": 0.9987
  }
}

---

**<FollowUp label="Would you like help pushing this to GitHub via Git commands?" query="Can you guide me through initializing Git and pushing this project to GitHub?"/>**
