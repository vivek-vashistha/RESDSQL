# Evaluate Spider2 with FastAPI

This script evaluates Spider2 questions by:
1. Reading questions from `spider2-lite.jsonl`
2. Generating SQL using the FastAPI server
3. Executing SQL queries on SQLite databases
4. Comparing execution results with golden CSV files from Spider2 repo

## Prerequisites

1. **FastAPI server must be running**:
   ```bash
   bash start_api.sh
   # Or: python api_server_refactored.py
   ```

2. **Required files**:
   - `spider2-lite.jsonl` - Questions dataset
   - Spider2 gold directory with:
     - `exec_result/` - Golden CSV result files
     - `spider2lite_eval.jsonl` - Evaluation metadata
   - SQLite database files in `./database/` directory

## Usage

### Basic Usage (SQLite only)

```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl /path/to/spider2-lite.jsonl \
  --gold_dir /path/to/Spider2/spider2-lite/evaluation_suite/gold
```

### Full Options

```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl /path/to/spider2-lite.jsonl \
  --gold_dir /path/to/Spider2/spider2-lite/evaluation_suite/gold \
  --api_url http://localhost:8000/infer \
  --tables_path ./data/spider2/converted/spider2-lite_tables.json \
  --database_dir ./database \
  --target_type natsql \
  --output evaluation_results.json
```

### Example with Your Setup

```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl /Users/vivekvashistha/Projects/Clients/Turing/Projects/Spider2/spider2-lite/spider2-lite.jsonl \
  --gold_dir /Users/vivekvashistha/Projects/Clients/Turing/Projects/Spider2/spider2-lite/evaluation_suite/gold \
  --output spider2_evaluation_results.json
```

### Handling Timeouts

If you encounter timeout errors, increase the timeout:

```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl /path/to/spider2-lite.jsonl \
  --gold_dir /path/to/gold \
  --api_timeout 600 \
  --max_retries 3
```

This sets:
- Timeout: 10 minutes (600 seconds)
- Retries: 3 attempts (with exponential backoff)

## Parameters

- `--spider2_jsonl`: Path to `spider2-lite.jsonl` file
- `--gold_dir`: Path to Spider2 gold directory (contains `exec_result/` and `spider2lite_eval.jsonl`)
- `--api_url`: FastAPI server URL (default: `http://localhost:8000/infer`)
- `--tables_path`: Path to `tables.json` file (default: `./data/spider2/converted/spider2-lite_tables.json`)
- `--database_dir`: Directory containing SQLite databases (default: `./database`)
- `--target_type`: Target SQL type - `"sql"` or `"natsql"` (default: `"natsql"`)
- `--output`: Optional path to save detailed results (JSON format)
- `--all_databases`: Evaluate all databases, not just SQLite (default: SQLite only)
- `--api_timeout`: API request timeout in seconds (default: 300 = 5 minutes)
- `--max_retries`: Maximum number of retries for failed API requests (default: 2)
- `--fail_on_error`: Stop evaluation on first error (default: skip and continue)

## Output

The script prints:
- Total examples evaluated
- Execution success rate (queries that executed successfully)
- Correct accuracy (queries that match golden results)
- Error count

If `--output` is specified, a detailed JSON file is saved with:
- Metrics summary
- Per-example results (SQL, execution status, correctness, errors)

## How It Works

1. **Load Questions**: Reads `spider2-lite.jsonl` to get questions and database IDs
2. **Filter SQLite**: By default, only evaluates examples with `local*` instance IDs (SQLite databases)
3. **Generate SQL**: For each question, calls FastAPI `/infer` endpoint to generate SQL
4. **Execute Query**: Runs the generated SQL on the corresponding SQLite database
5. **Compare Results**: Compares execution results with golden CSV files using:
   - `condition_cols` from evaluation metadata (if specified)
   - `ignore_order` flag from evaluation metadata
   - Proper handling of NaN values and floating-point tolerance

## Comparison Logic

The script uses the same comparison logic as the official Spider2 evaluation:
- Handles `condition_cols` (subset of columns to compare)
- Supports `ignore_order` (row order doesn't matter)
- Floating-point tolerance: 1e-2
- Proper NaN handling

## Troubleshooting

### "FastAPI server is not running"
- Start the server: `bash start_api.sh`

### "Database file not found"
- Ensure SQLite databases are in `./database/{db_id}/{db_id}.sqlite`
- Check that `db_id` in JSONL matches database directory names

### "Gold CSV not found"
- Verify `--gold_dir` points to the correct directory
- Ensure `exec_result/` subdirectory exists with CSV files

### "Failed to generate SQL via API" or Timeout Errors
- **Increase timeout**: Use `--api_timeout 600` (10 minutes) for complex questions
- **Check API server logs**: The server might be processing slowly
- **Verify `tables_path` is correct**: Ensure the path points to valid tables.json
- **Check that `db_id` exists in `tables.json`**: Some databases might be missing
- **Complex schemas**: Some questions with large schemas take longer; the script will retry automatically
- **Skip problematic questions**: By default, the script skips errors and continues; use `--fail_on_error` to stop on first error

## Performance

- **Per question**: ~2-5 seconds (API call + execution + comparison)
- **Total time**: Depends on number of SQLite examples (~250 examples = ~10-20 minutes)

## Example Output

```
Loading Spider2 data...
✓ FastAPI server is running

Evaluating 250 questions...
============================================================
Evaluating: 100%|████████████████████| 250/250 [15:23<00:00,  3.69s/it]

============================================================
Evaluation Results
============================================================
Total examples: 250
Execution success: 245
Execution success rate: 0.9800 (98.00%)
Execution correct: 180
Correct accuracy: 0.7200 (72.00%)
Errors: 5
============================================================
```
