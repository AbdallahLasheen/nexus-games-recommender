FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if required (like scikit-learn dependencies)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Command to run the Flask application
CMD ["python", "server.py"]
