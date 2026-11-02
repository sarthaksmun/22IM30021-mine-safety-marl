# Base Python image for Distributed Mine Safety MARL Platform
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Install system dependencies for OpenCV and GUI-free headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . /app

EXPOSE 8501

# Default entrypoint runs the full verification pipeline and launches the dashboard
CMD ["bash", "-c", "python run_all.py && streamlit run app.py --server.headless true --server.port 8501 --server.address 0.0.0.0"]
