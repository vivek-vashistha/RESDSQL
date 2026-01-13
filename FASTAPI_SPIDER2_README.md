# FastAPI Server with Spider2 Support

The FastAPI server (`api_server.py`) has been updated to support both Spider1 and Spider2 datasets. The FastAPI approach is **much faster** than batch preprocessing because:

1. **Models are loaded once** and cached in memory
2. **Database schemas are loaded once** and cached
3. **Database adapters are cached** (connection pooling for cloud databases)
4. **Preprocessing happens per-request** (one question at a time)

## Why FastAPI is Faster

The batch preprocessing script (`preprocessing.py`) is slow because it:
- Opens and closes database connections for every column query
- Processes all questions sequentially
- Re-loads models and schemas for each run

The FastAPI server avoids these issues by:
- Keeping connections open (connection pooling)
- Processing one question at a time (faster per-request)
- Caching models and schemas in memory

## Usage

### 1. Start the FastAPI Server

```bash
# For Spider1 (default)
python api_server.py

# Or use the start script
bash start_api.sh
```

The server will start on `http://localhost:8000` by default.

### 2. Make Inference Requests

#### For Spider1 (Original Spider Dataset)

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the names of all the countries?",
    "db_id": "world_1",
    "target_type": "natsql",
    "spider2_mode": false
  }'
```

#### For Spider2 (New Benchmark)

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the names of all the countries?",
    "db_id": "E_commerce",
    "target_type": "natsql",
    "spider2_mode": true,
    "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
  }'
```

### 3. Request Parameters

- `question` (required): Natural language question
- `db_id` (required): Database identifier
- `target_type` (optional): `"sql"` or `"natsql"` (default: `"natsql"`)
- `spider2_mode` (optional): Enable Spider2 mode (default: `false`)
- `tables_path` (optional): Path to `tables.json` file (default: `"./data/spider/tables.json"`)
- `use_contents` (optional): Use database contents for schema linking (default: `true`)
- `add_fk_info` (optional): Add foreign key information (default: `true`)
- `topk_table_num` (optional): Number of top tables to use (default: `4`)
- `topk_column_num` (optional): Number of top columns per table (default: `5`)
- `num_beams` (optional): Beam search size (default: `8`)
- `num_return_sequences` (optional): Number of sequences to return (default: `8`)

### 4. Response Format

```json
{
  "sql": "SELECT name FROM countries",
  "input_sequence": "...",
  "execution_success": true,
  "error": null
}
```

## Spider2 Setup

### For SQLite Databases

No additional setup needed. The server will automatically:
- Detect SQLite databases from `db_id` patterns (e.g., `E_commerce`, `Baseball`)
- Use the standard path: `database/{db_id}/{db_id}.sqlite`

### For Cloud Databases (BigQuery/Snowflake)

Set environment variables before starting the server:

#### BigQuery

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Snowflake

```bash
export SNOWFLAKE_USER="your-username"
export SNOWFLAKE_PASSWORD="your-password"
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_WAREHOUSE="your-warehouse"
export SNOWFLAKE_DATABASE="your-database"  # Optional, uses db_id if not set
export SNOWFLAKE_SCHEMA="PUBLIC"  # Optional, defaults to PUBLIC
```

## Database Type Detection

The server automatically detects database types from `db_id` patterns:

- `local*` or no prefix → SQLite
- `bq*` → BigQuery
- `sf*` or `snow*` → Snowflake

## Connection Pooling

The FastAPI server implements connection pooling for cloud databases:
- **SQLite**: Connections are created on-demand (fast enough for local files)
- **BigQuery/Snowflake**: Adapters are cached and reused across requests

This significantly speeds up repeated queries to the same database.

## Example: Testing with Spider2

1. **Convert Spider2 data** (if not already done):
   ```bash
   python utils/spider2_converter.py \
     --input data/spider2/spider2-lite.jsonl \
     --output_dir data/spider2/converted \
     --dataset_name spider2-lite
   ```

2. **Start the server**:
   ```bash
   python api_server.py
   ```

3. **Test with a question**:
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

## Performance Comparison

- **Batch preprocessing**: ~30-50 seconds per question (with many tables/columns)
- **FastAPI server**: ~2-5 seconds per question (after initial model load)

The FastAPI server is **10-25x faster** for individual queries because:
- Models are already loaded
- Schemas are cached
- Connections are pooled
- Only one question is processed at a time

## Troubleshooting

### "Database not found" error

Make sure:
1. The `tables_path` points to the correct `tables.json` file
2. The `db_id` exists in the `tables.json`
3. For SQLite, the database file exists at `database/{db_id}/{db_id}.sqlite`

### "Could not create adapter" warning

For cloud databases:
1. Check that environment variables are set correctly
2. Verify credentials are valid
3. Ensure network connectivity to cloud services

### Slow first request

The first request is slower because:
- Models need to be loaded (if not already cached)
- Database schemas need to be loaded
- This is a one-time cost per server restart

Subsequent requests are much faster.
