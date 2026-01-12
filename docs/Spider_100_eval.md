* RESDSQL Spider Dev (100 Examples) Evaluation Guide

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

### 8. Differences vs. API Pipeline (`/infer`)

The FastAPI server in `api_server.py` follows the same high-level idea (preprocess → schema classifier → ranked schema → T5 → decode), but there are important implementation differences compared to this authors' pipeline:

| **Step**                        | **Authors' offline pipeline (this doc)**                                                                                                                                                                        | **Current API `/infer` pipeline**                                                                                                                                                                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input data**                  | Uses `data/spider/dev_100.json` and creates a single preprocessed JSON (`preprocessed_test_natsql_100.json`) for all 100 examples at once.                                                                        | For each request, builds `preprocessed_data` on the fly via `preprocess_single_question` using `data/spider/tables.json` and the SQLite DB.                                                                                                              |
| **Schema classifier weights**   | `schema_item_classifier.py` loads fine‑tuned checkpoint `models/text2natsql_schema_item_classifier/dense_classifier.pt`.                                                                                         | API also loads the same `dense_classifier.pt`, but does not yet implement the full multi‑pass truncation‑recovery logic from `schema_item_classifier.py`.                                                                                                |
| **Classifier input formatting** | `ColumnAndTableClassifierDataset`: table names are plain (`table_name`), column infos are `column_name` or `column_name ( contents )`, with optional `[FK]` tags when `--add_fk_info` is used.            | API mirrors this: table names from `table_name`, column infos built as `column_name` or `column_name ( contents )`. For NatSQL, `add_fk_info` is disabled by default to match the offline setting.                                                     |
| **Top‑k schema selection**     | `text2sql_data_generator.py` (eval mode) uses `table_pred_probs` / `column_pred_probs` from the precomputed JSON to select top‑4 tables and top‑5 columns per table, writes `resdsql_test_natsql_100.json`. | API computes `table_probs` / `column_probs_per_table` in memory for a single example and selects top‑4 tables / top‑5 columns per table before calling `prepare_input_and_output`. No persisted `resdsql_test_natsql_100.json` is used.              |
| **NatSQL tables file**          | Uses `data/preprocessed_data/test_tables_for_natsql_100.json` created specifically for this 100‑example run.                                                                                                       | API currently uses the default `data/preprocessed_data/test_tables_for_natsql.json` for all requests.                                                                                                                                                        |
| **T5 + decode settings**        | `text2sql.py`: `batch_size=4`, `num_beams=4`, `num_return_sequences=4`, runs in batch over the 100 examples.                                                                                                  | API defaults:`num_beams=8`, `num_return_sequences=8` (configurable via request), runs **one example per HTTP call**.                                                                                                                                 |
| **Evaluation**                  | Uses `EvaluateTool` inside `text2sql.py` on the predictions in `predictions/Spider-dev-100/resdsql_base_natsql/pred.sql`, giving EM=0.76 / EX=0.91.                                                             | `scripts/test_api_spider_subset.py` calls `/infer` for each of the 100 examples and then uses the same `EvaluateTool` on the returned `sql`s`. Because the upstream steps differ (rows above), the EM/EX numbers are currently lower than 0.76 / 0.91. |

#### Key root causes of the current API accuracy gap

- **Different preprocessing source (biggest difference)**

  - **Authors**: all 100 examples are preprocessed once into `preprocessed_test_natsql_100.json` by `preprocessing.py`, and this exact JSON (including column ordering, contents, and normalization quirks) is what both the schema classifier and T5 model see through `resdsql_test_natsql_100.json`.
  - **API**: each `/infer` call rebuilds its own `preprocessed_data` directly from `tables.json` + SQLite via `preprocess_single_question`. Even small differences in how columns/contents are assembled change classifier scores → different top‑k tables/columns → different `input_sequence` than the one that produced EM=0.76 / EX=0.91.
- **Missing multi‑pass truncation recovery in the classifier (second biggest)**

  - **Authors**: `schema_item_classifier.py` detects when long inputs cause tables/columns to be truncated, runs a second pass on `truncated_dataset.json`, and merges the missing probabilities back in. This ensures every table/column ultimately gets a reasonable probability before ranking.
  - **API**: `classify_schema_items(...)` does a **single pass only**. If some schema tokens were dropped by the 512‑token limit, those tables/columns never get proper probabilities and may be excluded from the top‑k schema, even though the offline pipeline would have recovered them.
- **NatSQL tables configuration**

  - **Authors**: the 100‑example evaluation uses `data/preprocessed_data/test_tables_for_natsql_100.json`, generated in step 3 specifically for that subset.
  - **API**: uses `data/preprocessed_data/test_tables_for_natsql.json` (full‑dev version). Any mismatch here affects how `decode_natsqls` repairs NatSQL into executable SQL, impacting both EM and EX.
- **Generation settings and per‑example vs batch execution**

  - **Authors**: `text2sql.py` runs the 100 examples as a batch with `num_beams=4`, `num_return_sequences=4`, so decoding behaviour is tuned to that setting.
  - **API**: by default uses `num_beams=8`, `num_return_sequences=8` and generates **one question per HTTP call**. Combined with different schema ranking, this can lead the API to pick a different “first executable” SQL than the offline run for the same question.

In short, even though both pipelines now use the **same pretrained checkpoints** (schema classifier + T5), the API’s EM/EX on the 100‑example subset (~0.15 / 0.40) is lower mainly because its **preprocessing + schema ranking + NatSQL tables configuration** are still not 1:1 identical with the authors' offline pipeline described above.
