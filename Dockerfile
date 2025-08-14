# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Avoid cache issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y --no-install-recommends \
       git curl && rm -rf /var/lib/apt/lists/*

# Copy app source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit command to run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
