#!/bin/bash

# Script to set up the environment and run the FastAPI OCR API server on Amazon Linux

# --- Configuration ---
# IMPORTANT: Adjust PROJECT_DIR to the absolute path of your project directory on the EC2 instance.
# For example, if you cloned your project into the ec2-user's home directory:
PROJECT_DIR="/home/ec2-user/ocr-api" 

# Name of the Conda environment
CONDA_ENV_NAME="ocr-api-env"

# Log file for the server
LOG_FILE="server.logs"

echo "--- Starting OCR API Server Setup ---"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Error: Project directory $PROJECT_DIR not found."; exit 1; }
echo "Current directory: $(pwd)"

# --- Conda Setup ---
# Check if conda is installed and install if it's not
# This is useful for setting up on a fresh Amazon Linux EC2 instance
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Attempting to install Miniconda..."
    # Download and install Miniconda for Linux x86_64
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    
    # Add conda to PATH for the current script session
    # The installer will also add it to .bashrc for future sessions
    source "$HOME/miniconda/bin/activate"
    conda init bash
    
    echo "Miniconda installed. You may need to restart your shell or run 'source ~/.bashrc' for conda to be available in new terminals."
fi

# Ensure conda commands are available for the script
eval "$(conda shell.bash hook)"

# Check if the target conda environment exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating it from environment.yml..."
    if [ -f "environment.yml" ]; then
        conda env create -f environment.yml || { echo "Error: Failed to create conda environment from environment.yml."; exit 1; }
    else
        echo "Error: environment.yml not found in $PROJECT_DIR. Cannot create conda environment."; exit 1;
    fi
fi

# Activate the conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."; exit 1; }

# --- Server Launch ---
echo "Launching FastAPI server..."
echo "Logs will be written to $LOG_FILE"

# Run the server in the background with nohup and redirect stdout and stderr to a log file
nohup python main.py > "$LOG_FILE" 2>&1 &

echo "Server started in the background. PID: $!"
echo "--- Script Finished ---"