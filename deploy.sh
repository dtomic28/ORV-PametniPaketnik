#!/bin/bash

set -e  # Exit immediately if a command fails

echo "🛠️  Building Docker images..."
docker compose build

echo "🚀 Starting containers..."
docker compose up -d --remove-orphans

echo "🧹 Cleaning up unused Docker images..."
docker image prune -a -f

echo "✅ Deployment complete."
