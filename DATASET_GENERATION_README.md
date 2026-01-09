# Dataset Format Generation Script

This script converts Spider dataset format to the target format used for evaluation and testing.

## Format

The script generates datasets in the following format:

```json
[
    {
        "turns": [
            {
                "input": "What year is the movie \" The Imitation Game \" from ?",
                "expected_output": "SELECT release_year FROM movie WHERE title  =  \"The Imitation Game\";",
                "metadata": {
                    "instance_id": "0",
                    "database": "imdb",
                    "target_type": "sql"
                }
            }
        ]
    },
    ...
]
```

## Usage

### Basic Usage

Convert Spider dev set to SQL format:
```bash
python generate_dataset_format.py --input data/spider/dev.json --output dev_output.json
```

Convert Spider train set to SQL format:
```bash
python generate_dataset_format.py --input data/spider/train_spider.json --output train_output.json
```

### With NatSQL Target Type

Convert to NatSQL format (requires NatSQL dataset):
```bash
python generate_dataset_format.py --input data/spider/dev.json --output dev_natsql_output.json --target_type natsql
```

### With Custom Options

Limit the number of entries:
```bash
python generate_dataset_format.py --input data/spider/dev.json --output dev_output.json --limit 100
```

Custom starting index:
```bash
python generate_dataset_format.py --input data/spider/dev.json --output dev_output.json --start_index 1000
```

## Command Line Arguments

- `--input` (required): Path to input Spider dataset JSON file
  - Example: `data/spider/dev.json`, `data/spider/train_spider.json`
  
- `--output` (required): Path to output JSON file
  - Example: `output.json`, `data/converted/dev_output.json`
  
- `--target_type` (optional): Target type - "sql" or "natsql" (default: "sql")
  
- `--start_index` (optional): Starting index for instance_id (default: 0)
  
- `--limit` (optional): Maximum number of entries to process (default: all)

## Examples

### Convert Spider Dev Set
```bash
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output data/converted/spider_dev.json
```

### Convert Spider Train Set (First 1000 entries)
```bash
python generate_dataset_format.py \
    --input data/spider/train_spider.json \
    --output data/converted/spider_train_sample.json \
    --limit 1000
```

### Convert with NatSQL
```bash
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output data/converted/spider_dev_natsql.json \
    --target_type natsql
```

## Input Format

The script expects Spider dataset format with the following fields:
- `question`: Natural language question
- `query`: SQL query string
- `db_id`: Database identifier

## Output Format

Each entry in the output contains:
- `turns`: Array with one turn containing:
  - `input`: The natural language question
  - `expected_output`: The SQL query
  - `metadata`: Object with:
    - `instance_id`: Unique identifier (string)
    - `database`: Database name (from `db_id`)
    - `target_type`: "sql" or "natsql"

## Notes

- The script automatically creates output directories if they don't exist
- Progress is shown with a progress bar for large datasets
- Statistics are printed after conversion (total entries, unique databases, etc.)
- Entries with missing required fields are skipped with a warning
