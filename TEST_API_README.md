# Testing the Inference API

This directory contains scripts to test the RESDSQL inference API against expected SQL and NatSQL queries.

## Scripts

### 1. `test_inference_api.py` - Batch Testing

Main test script that can test multiple questions from a dataset file.

**Features:**
- Tests API against expected SQL/NatSQL queries
- Supports both SQL and NatSQL target types
- Normalizes queries for comparison (handles whitespace differences)
- Provides detailed statistics and accuracy metrics
- Can save results to JSON file
- Shows accuracy breakdown by database

**Usage:**

```bash
# Test with 10 examples using NatSQL
python3 test_inference_api.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --data-file "NatSQL/NatSQLv1_6/train_spider.json" \
    --target-type "natsql" \
    --limit 10 \
    --verbose

# Test with SQL target type
python3 test_inference_api.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --data-file "data/spider/train_spider.json" \
    --target-type "sql" \
    --limit 50

# Test and save results
python3 test_inference_api.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --data-file "NatSQL/NatSQLv1_6/train_spider.json" \
    --target-type "natsql" \
    --limit 100 \
    --output "test_results.json"
```

**Arguments:**
- `--api-url`: API endpoint URL (default: http://localhost:8000/infer)
- `--data-file`: Path to test data JSON file (default: NatSQL/NatSQLv1_6/train_spider.json)
- `--target-type`: "sql" or "natsql" (default: natsql)
- `--limit`: Maximum number of test cases (default: all)
- `--verbose`: Print detailed results for each test case
- `--delay`: Delay between API calls in seconds (default: 0.1)
- `--output`: Save detailed results to JSON file

### 2. `test_single_question.py` - Single Question Testing

Quick test script for testing a single question.

**Usage:**

```bash
# Test a single question
python3 test_single_question.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --question "What year is the movie The Imitation Game from ?" \
    --db-id "imdb" \
    --target-type "natsql"

# Test with expected query for comparison
python3 test_single_question.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --question "What are the names of documents that do not have any images?" \
    --db-id "document_management" \
    --target-type "natsql" \
    --expected "select documents.document_name from documents  where  except_  @.@ is document_sections_images.*"
```

## Test Data Format

The test data files should be JSON arrays where each object contains:
- `question`: Natural language question
- `db_id`: Database identifier
- `query`: Expected SQL query (for SQL target type)
- `NatSQL`: Expected NatSQL query (for NatSQL target type)

Example:
```json
{
  "db_id": "document_management",
  "query": "SELECT document_name FROM documents EXCEPT ...",
  "question": "What are the names of documents that do not have any images?",
  "NatSQL": "select documents.document_name from documents  where  except_  @.@ is document_sections_images.*"
}
```

## Installation

Make sure you have the `requests` library installed:

```bash
pip install requests
```

## Example Output

```
Loading test data from: NatSQL/NatSQLv1_6/train_spider.json
Loaded 10 test cases
Target type: natsql
API URL: http://98.80.127.45:8000/infer
================================================================================

[1/10] Testing: What are the names of documents that do not have any images?...
  ✓ MATCH

[2/10] Testing: What is the name of the document that has the most sections?...
  ✗ MISMATCH
    Expected: select documents.document_name from documents ...
    Generated: select documents.document_name from documents ...

================================================================================
TEST SUMMARY
================================================================================
Total test cases: 10
Successful API calls: 10
Failed API calls: 0
Matched queries: 8
Mismatched queries: 2
Accuracy: 80.00%
================================================================================
```

## Notes

- The script normalizes SQL queries before comparison to handle whitespace differences
- API calls have a default timeout of 30 seconds
- A small delay (0.1s) is added between requests to avoid overwhelming the API
- Results can be saved to a JSON file for further analysis
