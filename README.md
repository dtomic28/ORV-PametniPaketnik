# ORV Pametni Paketnik API

This project provides a GPU-accelerated Flask API for facial recognition using TensorFlow, OpenCV, and a custom face recognition pipeline. It is containerized using Docker and deployed via Docker Compose with GPU support.

---

## 📦 Features

* ✅ Built on `tensorflow/tensorflow:2.15.0-gpu`
* ✅ Asynchronous Flask API using Gunicorn + Gevent
* ✅ Full GPU acceleration using NVIDIA Container Toolkit
* ✅ Clean project structure for face recognition development
* ✅ Mounted volumes for persistent data
* ✅ One-command deployment using `deploy.sh`

---

## ⚛️ Deployment Instructions

### ✅ Prerequisites

Make sure you have the following installed and set up:

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

To verify that your GPU is accessible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

### ⚡ Quick Start (Recommended)

In the root of this repository, run:

```bash
./deploy.sh
```

> This script will:
>
> * Pull the latest code from the `main` branch
> * Build the Docker image with GPU support
> * Start (or restart) the API container

If the script isn't executable yet, first run:

```bash
chmod +x deploy.sh
```

---

## 🛠 Manual Deployment (Alternative)

You can also manually build and run the service:

```bash
docker compose build
docker compose up -d
```

The API will be accessible at:

```
http://localhost:5000
```

---

## 📁 Project Structure

```
ORV-PametniPaketnik/
├── api/                          # Flask API code (app:app)
├── face_recognition/            # Face recognition logic
│   ├── images/                  # Mounted: user face images
│   ├── models/                  # Mounted: trained face encodings
│   ├── temp_dataset/            # Mounted: temporary dataset for training
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Image definition
├── docker-compose.yml           # Container + GPU deployment config
└── deploy.sh                    # Deployment helper script
```

---

## 🔌 GPU Access Check

To confirm the container sees the GPU, run:

```bash
docker exec -it paketnik-api python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If configured correctly, it should list your available GPU(s).

---

## 📅 Updating the App

To update to the latest version and redeploy:

```bash
./deploy.sh
```

This will:

* Pull the latest Git changes
* Rebuild the container
* Restart it with the latest code

---

## 🪩 Cleanup

To stop and remove the container:

```bash
docker compose down
```

To rebuild everything from scratch:

```bash
docker compose down --volumes --remove-orphans
docker compose build --no-cache
docker compose up -d
```

To remove dangling images and free space:

```bash
docker system prune -a
```

---

## 💠 Troubleshooting

* **Permission denied on `deploy.sh`**:

  ```bash
  chmod +x deploy.sh
  ```

* **Docker can't access the GPU**:

  * Ensure you have the NVIDIA Container Toolkit installed.
  * Make sure `nvidia-smi` works inside a test container.

* **Build fails due to space issues**:

  * Run `docker system prune -a` to clean up unused data.
  * Check Docker Desktop’s disk size limit (if on Windows).

---
