# Base image
FROM python:3.11-slim

# Metadata
LABEL maintainer="GitHub Copilot"

# Set working directory
WORKDIR /app

# Install system deps required by some packages (e.g., xgboost needs libgomp)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code and model files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Use a non-root user (optional for better security)
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

# Launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
