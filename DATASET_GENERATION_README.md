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

### Generate Both SQL and NatSQL Versions

Generate both SQL and NatSQL versions in separate files automatically:
```bash
python generate_dataset_format.py --input data/spider/dev.json --output dev_output.json --generate_both
```

This will create two files:
- `dev_output_sql.json` - SQL version
- `dev_output_natsql.json` - NatSQL version

The script will automatically try to find the corresponding NatSQL file if not specified with `--natsql_input`.

You can combine `--generate_both` with other options like `--limit` and `--start_index`:
```bash
# Generate first 100 entries in both formats
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output dev_output.json \
    --generate_both \
    --limit 100

# Generate entries starting from index 1000, limit 500
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output dev_output.json \
    --generate_both \
    --start_index 1000 \
    --limit 500
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
  
- `--natsql_input` (optional): Path to NatSQL dataset JSON file. If not provided and `--target_type natsql` or `--generate_both` is used, the script will try to auto-detect the NatSQL file.
  
- `--generate_both` (optional): Generate both SQL and NatSQL versions automatically in separate files
  
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

### Generate Both SQL and NatSQL Versions
```bash
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output data/converted/spider_dev.json \
    --generate_both
```

This will generate:
- `data/converted/spider_dev_sql.json` - SQL format
- `data/converted/spider_dev_natsql.json` - NatSQL format

### Generate Both Versions with Limit
```bash
python generate_dataset_format.py \
    --input data/spider/dev.json \
    --output data/converted/spider_dev.json \
    --generate_both \
    --limit 1000
```

This will generate both SQL and NatSQL versions for the first 1000 entries.

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
- When using `--generate_both`, the script automatically tries to find the NatSQL file in common locations:
  - Same directory as input with `-natsql` suffix
  - `NatSQL/NatSQLv1_6/` directory
  - `data/preprocessed_data/` directory
- If NatSQL data is not available for an entry, that entry will be skipped in the NatSQL output file
