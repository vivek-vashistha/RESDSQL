## RESDSQL Spider Dev (100 Examples) Evaluation Guide

This document records the exact commands and sample outputs to run the **official RESDSQL NatSQL evaluation** on a **100‑example subset** of Spider dev, using the **pretrained base checkpoints** only (no training).

All commands below are intended to be run **inside WSL**.

### 1. Activate environment and go to project

```bash
cd /mnt/d/Work/turing/RESDSQL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate resdsql
```

Expected: your shell prompt starts with `(resdsql)`.

### 2. Create a 100‑example Spider dev subset

This script copies the **first 100 entries** from `data/spider/dev.json` into `data/spider/dev_100.json`.

```bash
python scripts/create_spider_subset.py \
  --input data/spider/dev.json \
  --output data/spider/dev_100.json \
  --limit 100
```

Sample output (truncated):

```text
Wrote 100 examples to data/spider/dev_100.json
```

### 3. Prepare NatSQL tables file for this run

```bash
python NatSQL/table_transform.py \
  --in_file data/spider/tables.json \
  --out_file data/preprocessed_data/test_tables_for_natsql_100.json \
  --correct_col_type \
  --remove_start_table \
  --analyse_same_column \
  --table_transform \
  --correct_primary_keys \
  --use_extra_col_types \
  --db_path database
```

Output is mostly logging about type corrections; no EM/EX here.

### 4. Preprocess the 100 examples (NatSQL mode)

```bash
python preprocessing.py \
  --mode test \
  --table_path data/spider/tables.json \
  --input_dataset_path data/spider/dev_100.json \
  --output_dataset_path data/preprocessed_data/preprocessed_test_natsql_100.json \
  --db_path database \
  --target_type natsql
```

Sample output (progress bar):

```text
100it [00:01, 54.13it/s]
```

### 5. Run schema item classifier on these 100 examples

Uses the pretrained NatSQL classifier checkpoint in `models/text2natsql_schema_item_classifier/`.

```bash
python schema_item_classifier.py \
  --batch_size 32 \
  --device cpu \
  --seed 42 \
  --save_path ./models/text2natsql_schema_item_classifier \
  --dev_filepath data/preprocessed_data/preprocessed_test_natsql_100.json \
  --output_filepath data/preprocessed_data/test_with_probs_natsql_100.json \
  --use_contents \
  --mode test
```

Sample output (truncated):

```text
100%|████████████████████████████████████████████████████████████████████████| 4/4 [06:08<00:00, 92.15s/it]
```

### 6. Generate RESDSQL NatSQL test set (100 examples)

```bash
python text2sql_data_generator.py \
  --input_dataset_path data/preprocessed_data/test_with_probs_natsql_100.json \
  --output_dataset_path data/preprocessed_data/resdsql_test_natsql_100.json \
  --topk_table_num 4 \
  --topk_column_num 5 \
  --mode test \
  --use_contents \
  --output_skeleton \
  --target_type natsql
```

This step prints minimal log output; it just writes the new file
`data/preprocessed_data/resdsql_test_natsql_100.json`.

### 7. Run pretrained NatSQL base model and evaluate (EM / EXEC)

This uses the downloaded checkpoint:
`models/text2natsql-t5-base/checkpoint-14352/`.

```bash
mkdir -p predictions/Spider-dev-100/resdsql_base_natsql

python text2sql.py \
  --batch_size 4 \
  --device cpu \
  --seed 42 \
  --save_path ./models/text2natsql-t5-base/checkpoint-14352 \
  --mode eval \
  --dev_filepath data/preprocessed_data/resdsql_test_natsql_100.json \
  --original_dev_filepath data/spider/dev_100.json \
  --db_path ./database \
  --tables_for_natsql data/preprocessed_data/test_tables_for_natsql_100.json \
  --num_beams 4 \
  --num_return_sequences 4 \
  --target_type natsql \
  --output predictions/Spider-dev-100/resdsql_base_natsql/pred.sql
```

Key parts of the output (truncated):

```text
Namespace(batch_size=4, db_path='./database', dev_filepath='data/preprocessed_data/resdsql_test_natsql_100.json', device='cpu', ...
...
100%|████████████████████████████████████████████████████████████████████████| 25/25 [12:26<00:00, 29.84s/it]
Text-to-SQL inference spends 763.6128602027893s.
exact_match score: 0.76
exec score: 0.91
```

- **exact_match score 0.76** → 76% EM on these 100 examples.  
- **exec score 0.91** → 91% EX on these 100 examples.

The predicted SQLs are saved in:

```text
predictions/Spider-dev-100/resdsql_base_natsql/pred.sql
```

You can re-run this entire evaluation in the future by executing the commands in sections **1–7** in order.


