#!/bin/bash

# Stop RESDSQL FastAPI Server
# Usage: ./stop_api.sh [port]

set -e

PORT=${1:-8000}
PID_FILE="logs/api_server_${PORT}.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE"
    echo "Server may not be running on port $PORT"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID is not running"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Stopping server with PID: $PID"
kill "$PID"

# Wait a bit and check if it's still running
sleep 2
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Process still running, forcing kill..."
    kill -9 "$PID"
fi

rm -f "$PID_FILE"
echo "Server stopped successfully"
