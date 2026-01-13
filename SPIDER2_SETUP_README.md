# Spider2 Evaluation Setup Guide

This guide explains how to set up and use RESDSQL for evaluating on the Spider2 dataset.

## Overview

Spider2 is a new benchmark that evaluates text-to-SQL models on real-world enterprise databases:
- **Spider2-Lite**: 547 examples across BigQuery, Snowflake, and SQLite
- **Spider2-Snow**: 547 examples on Snowflake only

## Prerequisites

1. **Install additional dependencies**:
   ```bash
   pip install snowflake-connector-python>=3.0.0
   pip install google-cloud-bigquery>=3.0.0
   ```

2. **Download Spider2 dataset**:
   - Clone the [Spider2 repository](https://github.com/xlang-ai/Spider2)
   - Copy `spider2-lite.jsonl` and/or `spider2-snow.jsonl` to `data/spider2/`

3. **Set up cloud database access** (if using Spider2-Lite or Spider2-Snow):
   
   **BigQuery** (for Spider2-Lite):
   - Follow the [BigQuery setup guide](https://github.com/xlang-ai/Spider2)
   - Set environment variable: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"`
   
   **Snowflake** (for Spider2-Snow and Spider2-Lite):
   - Fill out the [Spider2 Snowflake Access form](https://github.com/xlang-ai/Spider2)
   - Set environment variables:
     ```bash
     export SNOWFLAKE_USER="your_username"
     export SNOWFLAKE_PASSWORD="your_password"
     export SNOWFLAKE_ACCOUNT="your_account"
     export SNOWFLAKE_WAREHOUSE="your_warehouse"
     export SNOWFLAKE_DATABASE="your_database"
     export SNOWFLAKE_SCHEMA="your_schema"
     ```

## Quick Start

### Step 1: Convert Spider2 Data

First, convert the Spider2 JSONL format to RESDSQL format:

```bash
python utils/spider2_converter.py \
    --input data/spider2/spider2-lite.jsonl \
    --output_dir data/spider2/converted \
    --dataset_name spider2-lite
```

This will create:
- `data/spider2/converted/spider2-lite_tables.json` - Schema information
- `data/spider2/converted/spider2-lite_dataset.json` - Converted dataset

### Step 2: Run Inference

Run inference using the Spider2-specific script:

```bash
# For Spider2-Lite
sh scripts/inference/infer_text2natsql_spider2.sh base lite

# For Spider2-Snow
sh scripts/inference/infer_text2natsql_spider2.sh base snow

# With custom input file
sh scripts/inference/infer_text2natsql_spider2.sh base lite data/spider2/custom.jsonl
```

The script will:
1. Convert Spider2 JSONL to RESDSQL format (if not already done)
2. Preprocess the dataset
3. Classify schema items
4. Generate SQL predictions

Predictions will be saved to: `predictions/{dataset_name}/resdsql_{model_scale}_natsql/pred.sql`

### Step 3: Evaluate Predictions

Evaluate the predictions:

```bash
python evaluate_spider2.py \
    --predictions predictions/spider2-lite/resdsql_base_natsql/pred.sql \
    --spider2_data data/spider2/spider2-lite.jsonl \
    --output evaluation_results.json
```

If you have gold SQL:
```bash
python evaluate_spider2.py \
    --predictions predictions/spider2-lite/resdsql_base_natsql/pred.sql \
    --gold path/to/gold_sql.txt \
    --spider2_data data/spider2/spider2-lite.jsonl \
    --output evaluation_results.json
```

## Model Scales

You can use different model scales:
- `base` - T5-Base (recommended for 16GB RAM)
- `large` - T5-Large
- `3b` - T5-3B (requires more memory)

Example:
```bash
sh scripts/inference/infer_text2natsql_spider2.sh large lite
```

## Testing with SQLite Subset

To test with only SQLite databases (no cloud setup required):

1. Filter the Spider2-Lite JSONL to only SQLite examples
2. Run the inference script on the filtered file
3. The system will automatically detect SQLite databases

## File Structure

After setup, your directory structure should look like:

```
RESDSQL/
├── data/
│   └── spider2/
│       ├── spider2-lite.jsonl
│       ├── spider2-snow.jsonl
│       ├── converted/
│       │   ├── spider2-lite_tables.json
│       │   └── spider2-lite_dataset.json
│       └── README.md
├── predictions/
│   └── spider2-lite/
│       └── resdsql_base_natsql/
│           └── pred.sql
├── utils/
│   ├── database_adapters.py
│   └── spider2_converter.py
├── scripts/
│   └── inference/
│       └── infer_text2natsql_spider2.sh
└── evaluate_spider2.py
```

## Troubleshooting

### Database Connection Errors

- **Snowflake**: Ensure all environment variables are set correctly
- **BigQuery**: Verify credentials file path and permissions
- **SQLite**: Check that database files exist in the expected location

### Schema Extraction Failures

If schema extraction fails for cloud databases:
- Check network connectivity
- Verify database credentials
- Ensure you have read permissions on the databases

### Memory Issues

If you encounter out-of-memory errors:
- Use the `base` model instead of `large` or `3b`
- Reduce batch size in the inference script
- Process in smaller batches

## Notes

- RESDSQL models trained on original Spider should work without retraining
- Performance may vary on enterprise databases due to different SQL dialects
- Cloud database queries may incur costs (especially BigQuery)
- The evaluation script executes queries on actual databases, so ensure you have proper access

## References

- [Spider2 Repository](https://github.com/xlang-ai/Spider2)
- [Spider2 Paper](https://arxiv.org/abs/2411.07763)
- [Spider2 Website](https://spider2-sql.github.io/)
