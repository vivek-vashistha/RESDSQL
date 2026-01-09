#!/bin/bash

# Start RESDSQL FastAPI Server
# Usage: ./start_api.sh [port] [host]

set -e

# Default values
PORT=${1:-8000}
HOST=${2:-0.0.0.0}

# Use conda Python if available, otherwise use system Python
if [ -f "/opt/anaconda3/envs/resdsql/bin/python" ]; then
    PYTHON="/opt/anaconda3/envs/resdsql/bin/python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python"
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    if conda env list | grep -q "resdsql"; then
        echo "Activating conda environment: resdsql"
        eval "$(conda shell.bash hook)"
        conda activate resdsql
    fi
fi

# Set device to CPU for Mac (change to "mps" for Apple Silicon GPU, or "0" for CUDA GPU)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use CPU or MPS
        export CUDA_VISIBLE_DEVICES=""
        echo "Running on macOS - using CPU (set device='mps' in api_server.py for Apple Silicon GPU)"
    else
        # Linux - try to use GPU if available
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU detected - using CUDA"
        else
            export CUDA_VISIBLE_DEVICES=""
            echo "No GPU detected - using CPU"
        fi
    fi
fi

echo "Starting RESDSQL API server on http://${HOST}:${PORT}"
echo "API documentation available at: http://${HOST}:${PORT}/docs"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
$PYTHON -m uvicorn api_server:app --host $HOST --port $PORT
