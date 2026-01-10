# Evaluation Optimization Guide

This guide explains how to use the optimized evaluation script that significantly speeds up evaluation by using connection pooling and parallel processing.

## Overview

The optimized evaluation script (`evaluate_text2sql_ckpts_optimized.py`) provides:
- **Connection pooling**: Keeps SQLite connections open and reuses them
- **Parallel processing**: Evaluates multiple predictions concurrently
- **Thread-safe operations**: Safe concurrent database access

## Expected Performance Improvement

- **3-5x faster** with 4-8 workers on multi-core systems
- **Reduced I/O overhead** from connection pooling
- **Better CPU utilization** from parallel processing

## Usage

**Important:** Make sure you're in the project root directory (e.g., `~/RESDSQL` or wherever you cloned/uploaded the project) before running any commands.

### Basic Usage

Run the optimized evaluation script with default settings (4 workers):

```bash
# Navigate to project directory first (if not already there)
cd ~/RESDSQL  # or your project path

python evaluate_text2sql_ckpts_optimized.py \
    --save_path "./models/text2natsql-t5-base" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/test_tables_for_natsql.json" \
    --target_type "natsql"
```

### Customize Number of Workers

For EC2 machines with more CPU cores, increase the number of workers:

```bash
python evaluate_text2sql_ckpts_optimized.py \
    --save_path "./models/text2natsql-t5-base" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/test_tables_for_natsql.json" \
    --target_type "natsql" \
    --num_workers 8
```

### Disable Parallel Processing

If you want to use the original sequential evaluation:

```bash
python evaluate_text2sql_ckpts_optimized.py \
    --save_path "./models/text2natsql-t5-base" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/test_tables_for_natsql.json" \
    --target_type "natsql" \
    --num_workers 1 \
    --no-use_parallel
```

## Recommended Settings for EC2

### For EC2 instances with 4-8 CPU cores:
```bash
--num_workers 4
```

### For EC2 instances with 8-16 CPU cores:
```bash
--num_workers 8
```

### For EC2 instances with 16+ CPU cores:
```bash
--num_workers 16
```

**Note**: Don't set `num_workers` higher than your CPU core count. Too many workers can actually slow things down due to context switching overhead.

## All Arguments

The optimized script supports all the same arguments as the original, plus:

- `--num_workers`: Number of parallel workers (default: 4)
- `--use_parallel`: Enable parallel processing (default: True)
- `--no-use_parallel`: Disable parallel processing

## Comparison with Original Script

| Feature | Original (`evaluate_text2sql_ckpts.py`) | Optimized (`evaluate_text2sql_ckpts_optimized.py`) |
|---------|------------------------------------------|-----------------------------------------------------|
| Connection Management | Opens/closes for each query | Connection pooling |
| Processing | Sequential | Parallel (configurable) |
| Speed | Baseline | 3-5x faster |
| CPU Usage | Single core | Multi-core |

## Troubleshooting

### If you get "Too many open files" error:
- Reduce `--num_workers` to a lower number (e.g., 2-4)
- This happens when too many database connections are opened

### If evaluation is slower than expected:
- Check CPU usage with `htop` or `top`
- Make sure `num_workers` doesn't exceed CPU cores
- Try disabling parallel processing to compare: `--no-use_parallel`

### If you get SQLite locking errors:
- The connection pool handles this automatically
- If issues persist, reduce `--num_workers`

## Example: Full Evaluation Run

**Important:** Make sure you're in the project root directory (e.g., `~/RESDSQL` or wherever you cloned/uploaded the project) before running the command.

```bash
# Navigate to project directory first (if not already there)
cd ~/RESDSQL  # or your project path

# Evaluate all checkpoints with 8 workers
python evaluate_text2sql_ckpts_optimized.py \
    --batch_size 8 \
    --device "0" \
    --save_path "./models/text2natsql-t5-base" \
    --eval_results_path "./eval_results/text2natsql-t5-base-optimized" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/test_tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --num_workers 8
```

## Performance Monitoring

The script will print:
- Inference time (model generation)
- Evaluation time (with optimization)
- Progress updates every 100 evaluations

Example output:
```
Text-to-SQL inference spends 45.23s.
Starting evaluation with 8 workers...
Processed 100/1034 evaluations
Processed 200/1034 evaluations
...
exact_match score: 0.741
exec score: 0.802
Evaluation time: 120.45s (optimized)
```

Compare this with the original script to see the speedup!
