#!/bin/bash

# Start RESDSQL FastAPI Server in the background (Refactored version that strictly reuses all components)
# Usage: ./start_api_background.sh [port] [host]

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
        echo "Running on macOS - using CPU (set device='mps' in api_server_refactored.py for Apple Silicon GPU)"
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file paths
LOG_FILE="logs/api_server_${PORT}.log"
PID_FILE="logs/api_server_${PORT}.pid"

echo "Starting RESDSQL API server (refactored) in background on http://${HOST}:${PORT}"
echo "Logs will be written to: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo ""

# Check if server is already running on this port
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Server is already running with PID $OLD_PID"
        echo "To stop it, run: kill $OLD_PID"
        echo "Or use: ./stop_api.sh $PORT"
        exit 1
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Start the server in background with nohup (using refactored implementation that strictly reuses all components)
cd "$(dirname "$0")"
nohup $PYTHON -m uvicorn api_server_refactored:app --host $HOST --port $PORT > "$LOG_FILE" 2>&1 &

# Save the PID
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

echo "Server started with PID: $SERVER_PID"
echo "To view logs: tail -f $LOG_FILE"
echo "To stop the server: kill $SERVER_PID"
echo "Or use: ./stop_api.sh $PORT"
echo ""
echo "Server is running in the background and will continue even if you close this terminal."
