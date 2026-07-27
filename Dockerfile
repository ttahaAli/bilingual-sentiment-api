# 1. Start with a lightweight, official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app
ENV PYTHONPATH=/app

# 3. Copy only requirements first to leverage Docker's caching layers
COPY requirements.txt .

# 4. First install standard libraries from standard PyPI, then pull CPU PyTorch specifically
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cpu

# 5. Copy the rest of your application code and model weights into the container
COPY . .

# 6. Expose port 8000 for the API
EXPOSE 8000

# 7. Tell Docker how to run the FastAPI app on startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]