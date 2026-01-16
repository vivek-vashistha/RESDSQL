# Quick Start: Spider2 FastAPI Server

Use the FastAPI server to ask questions one by one instead of running the full batch inference script.

## ✅ What You Already Have

Based on your setup, you have:
- ✅ `data/spider2/converted/spider2-lite_tables.json` - Schema information
- ✅ `data/preprocessed_data/test_tables_for_natsql_spider2.json` - NatSQL tables (from Step 2)
- ✅ SQLite database files in `database/` directory

## 🚀 Quick Start

### 1. Start the FastAPI Server

```bash
# Option 1: Use the start script
bash start_api.sh

# Option 2: Direct command
python api_server_refactored.py
# Or with uvicorn
uvicorn api_server_refactored:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### 2. Ask Questions (One by One)

#### Using curl:

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many orders are there?",
    "db_id": "E_commerce",
    "target_type": "natsql",
    "spider2_mode": true,
    "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/infer",
    json={
        "question": "How many orders are there?",
        "db_id": "E_commerce",
        "target_type": "natsql",
        "spider2_mode": true,
        "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
    }
)

result = response.json()
print(f"SQL: {result['sql']}")
```

#### Using the API docs (Interactive):

1. Open browser: `http://localhost:8000/docs`
2. Click on `/infer` endpoint
3. Click "Try it out"
4. Fill in the request body
5. Click "Execute"

## 📋 Request Parameters

- **question** (required): Your natural language question
- **db_id** (required): Database ID from Spider2 (e.g., "E_commerce", "Baseball")
- **target_type** (optional): `"sql"` or `"natsql"` (default: `"natsql"`)
- **spider2_mode** (required): Set to `true` for Spider2
- **tables_path** (optional): Path to tables.json (defaults to Spider1 path if not provided)

## 📝 Example Questions from Spider2

You can test with questions from `spider2-lite.jsonl`:

```bash
# Example 1: E_commerce database
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "According to the RFM definition document, calculate the average sales per order for each customer within distinct RFM segments, considering only delivered orders.",
    "db_id": "E_commerce",
    "target_type": "natsql",
    "spider2_mode": true,
    "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
  }'

# Example 2: Baseball database
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Could you help me calculate the average single career span value in years for all baseball players?",
    "db_id": "Baseball",
    "target_type": "natsql",
    "spider2_mode": true,
    "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
  }'
```

## 🎯 Response Format

```json
{
  "sql": "SELECT COUNT(*) FROM orders",
  "input_sequence": "How many orders are there? | orders : orders.order_id, orders.customer_id, ...",
  "execution_success": true,
  "error": null
}
```

## ⚡ Why FastAPI is Better

- **Faster**: Models loaded once, cached in memory
- **Interactive**: Ask questions one by one
- **No batch processing**: Skip the slow preprocessing steps
- **Real-time**: Get SQL answers immediately

## 🔍 Find Database IDs

To find available `db_id` values, check:
```bash
python -c "import json; data = json.load(open('data/spider2/converted/spider2-lite_tables.json')); print([d['db_id'] for d in data])"
```

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
