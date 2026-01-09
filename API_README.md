# RESDSQL FastAPI Server

This document describes how to use the FastAPI server for RESDSQL Text-to-SQL inference.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements now include:
- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`
- `pydantic==2.5.0`

## Starting the Server

### Foreground (Interactive) Mode

**Basic Usage:**
```bash
python api_server.py
```

**Using the startup script:**
```bash
./start_api.sh
```

**Using Uvicorn Directly:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The server will start on `http://0.0.0.0:8000` by default. This mode runs in the foreground and will stop when you close the terminal.

### Background Mode (Persistent)

To run the server in the background so it continues even after closing the terminal:

**Using the background script:**
```bash
./start_api_background.sh [port] [host]
```

Example:
```bash
./start_api_background.sh 8000 0.0.0.0
```

This will:
- Start the server in the background using `nohup`
- Create a PID file at `logs/api_server_<port>.pid`
- Write logs to `logs/api_server_<port>.log`
- Continue running even after you close the terminal

**Stopping the background server:**
```bash
./stop_api.sh [port]
```

Example:
```bash
./stop_api.sh 8000
```

**Viewing logs:**
```bash
tail -f logs/api_server_8000.log
```

### Alternative Background Methods

**Using nohup directly:**
```bash
nohup uvicorn api_server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
```

**Using screen (recommended for development):**
```bash
screen -S resdsql_api
python api_server.py
# Press Ctrl+A then D to detach
# Reattach with: screen -r resdsql_api
```

**Using tmux:**
```bash
tmux new-session -d -s resdsql_api 'python api_server.py'
# Attach with: tmux attach -t resdsql_api
# Detach with: Ctrl+B then D
```

### With Custom Configuration

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1 --reload
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the server is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

### 2. Root Endpoint

**GET** `/`

Get API information and available endpoints.

### 3. SQL Inference

**POST** `/infer`

Generate SQL from a natural language question.

**Request Body:**
```json
{
  "question": "What are the names of all students?",
  "db_id": "student",
  "target_type": "natsql",
  "use_contents": true,
  "add_fk_info": true,
  "topk_table_num": 4,
  "topk_column_num": 5,
  "num_beams": 8,
  "num_return_sequences": 8
}
```

**Parameters:**
- `question` (required): Natural language question
- `db_id` (required): Database identifier (must exist in `./data/spider/tables.json`)
- `target_type` (optional): `"sql"` or `"natsql"` (default: `"natsql"`)
- `use_contents` (optional): Whether to use database contents (default: `true`)
- `add_fk_info` (optional): Whether to add foreign key information (default: `true`)
- `topk_table_num` (optional): Number of top tables to consider (default: `4`)
- `topk_column_num` (optional): Number of top columns per table (default: `5`)
- `num_beams` (optional): Beam search size (default: `8`)
- `num_return_sequences` (optional): Number of sequences to return (default: `8`)

**Response:**
```json
{
  "sql": "SELECT name FROM student",
  "input_sequence": "What are the names of all students? | student : student.name",
  "execution_success": true,
  "error": null
}
```

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the names of all students?",
    "db_id": "student",
    "target_type": "natsql"
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/infer"
data = {
    "question": "What are the names of all students?",
    "db_id": "student",
    "target_type": "natsql"
}

response = requests.post(url, json=data)
result = response.json()
print(result["sql"])
```

### Using JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

const url = 'http://localhost:8000/infer';
const data = {
  question: 'What are the names of all students?',
  db_id: 'student',
  target_type: 'natsql'
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
  .then(res => res.json())
  .then(result => console.log(result.sql));
```

## Model Loading

The server automatically loads models on startup. By default, it loads:
- Schema classifier: `./models/text2natsql_schema_item_classifier` (for NatSQL) or `./models/text2sql_schema_item_classifier` (for SQL)
- Text2SQL model: `./models/text2natsql-t5-base/checkpoint-14352` (for NatSQL) or `./models/text2sql-t5-base/checkpoint-39312` (for SQL)

### Model Paths

The server looks for models in the following locations:

**For NatSQL:**
- Schema classifier: `./models/text2natsql_schema_item_classifier`
- T5-Base: `./models/text2natsql-t5-base/checkpoint-14352`
- T5-Large: `./models/text2natsql-t5-large/checkpoint-21216`
- T5-3B: `./models/text2natsql-t5-3b/checkpoint-78302`

**For SQL:**
- Schema classifier: `./models/text2sql_schema_item_classifier`
- T5-Base: `./models/text2sql-t5-base/checkpoint-39312`
- T5-Large: `./models/text2sql-t5-large/checkpoint-30576`
- T5-3B: `./models/text2sql-t5-3b/checkpoint-78302`

## Required Files

Make sure you have the following files/directories:
- `./data/spider/tables.json` - Database schema information
- `./database/` - Directory containing SQLite database files
- Model checkpoints in `./models/` directory

## Device Configuration

The server automatically detects and uses the best available device:
- CUDA GPU (if available)
- Apple Silicon GPU (MPS, if available)
- CPU (fallback)

You can force CPU usage by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `500`: Server error (model loading failure, inference error, etc.)

Error responses include a `detail` field with error information:
```json
{
  "detail": "Database student not found in tables.json"
}
```

## Performance Considerations

1. **Model Loading**: Models are loaded once at startup. The first request may be slower if models need to be loaded.

2. **Batch Processing**: The API processes one question at a time. For batch processing, make multiple requests or modify the code to handle batches.

3. **Memory**: Ensure sufficient RAM/VRAM for the model size:
   - T5-Base: ~2-3GB RAM
   - T5-Large: ~5-6GB RAM
   - T5-3B: ~14-16GB RAM

4. **Device**: GPU inference is significantly faster than CPU. Use GPU when available.

## Running as a System Service

### macOS (using launchd)

Create a plist file at `~/Library/LaunchAgents/com.resdsql.api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.resdsql.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>api_server:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/RESDSQL</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/path/to/RESDSQL/logs/api_server.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/RESDSQL/logs/api_server.error.log</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.resdsql.api.plist
launchctl start com.resdsql.api
```

### Linux (using systemd)

Create a service file at `/etc/systemd/system/resdsql-api.service`:

```ini
[Unit]
Description=RESDSQL FastAPI Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/RESDSQL
Environment="PATH=/path/to/conda/envs/resdsql/bin"
ExecStart=/path/to/conda/envs/resdsql/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable resdsql-api
sudo systemctl start resdsql-api
sudo systemctl status resdsql-api
```

## Troubleshooting

### Models Not Loading

If models fail to load:
1. Check that model checkpoints exist in the expected paths
2. Verify file permissions
3. Check available memory/disk space
4. Review error logs in the console

### Database Not Found

If you get "Database not found" errors:
1. Verify `db_id` exists in `./data/spider/tables.json`
2. Check that the database file exists at `./database/{db_id}/{db_id}.sqlite`

### Out of Memory

If you encounter OOM errors:
1. Use a smaller model (T5-Base instead of T5-Large/3B)
2. Reduce `num_beams` and `num_return_sequences`
3. Use CPU instead of GPU
4. Close other applications to free memory

## Integration Examples

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
RESDSQL_API = "http://localhost:8000/infer"

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    response = requests.post(RESDSQL_API, json=data)
    return jsonify(response.json())
```

### Docker Deployment

```dockerfile
FROM python:3.8

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

Same as the main RESDSQL project.
