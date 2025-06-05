# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.15.0-gpu

# Install required system packages for OpenCV (libGL, etc.)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /app

# Copy all necessary source code
COPY face_recognition/ /app/face_recognition/
COPY api/ /app/api/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r /app/requirements.txt
RUN pip install gevent  # required for async workers in gunicorn

# Expose port for Flask app
EXPOSE 5000

# Set working directory to /app/api to match import assumptions
WORKDIR /app/api

# Launch the app via Gunicorn with async workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--worker-class", "gevent", "app:app"]
