# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Thêm dòng này trước CMD
ENV PYTHONPATH=/app
# # Set environment variables
# ENV MODEL_PATH=/models/transaction_classifier.pkl

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]  