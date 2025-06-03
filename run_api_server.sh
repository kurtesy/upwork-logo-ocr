#!/bin/bash

# Script to run the FastAPI OCR API server

# --- Configuration ---
# IMPORTANT: Adjust PROJECT_DIR to the absolute path of your project directory on the EC2 instance.
# For example, if you cloned your project into the ec2-user's home directory:
PROJECT_DIR="/home/ec2-user/ocr-api" 

# Name of the virtual environment directory
VENV_DIR="venv"

echo "Starting OCR API Server..."

cd "$PROJECT_DIR" || { echo "Error: Project directory $PROJECT_DIR not found."; exit 1; }

echo "Current directory: $(pwd)"

if [ -d "$VENV_DIR" ]; then
    echo "Activating Python virtual environment..."
    source "$VENV_DIR/bin/activate" || { echo "Error: Failed to activate virtual environment."; exit 1; }
else
    echo "Warning: Virtual environment directory '$VENV_DIR' not found. Running with system Python."
fi

echo "Launching Uvicorn server with main.py..."
python main.py