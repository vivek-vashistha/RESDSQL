#!/bin/bash
set -e

device="0"
num_workers=8  # Adjust based on your EC2 instance CPU cores

python -u evaluate_text2sql_ckpts_optimized.py \
    --batch_size 3 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2natsql-t5-3b" \
    --eval_results_path "./eval_results/text2natsql-t5-3b-spider-dk-optimized" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_spider_dk_natsql.json" \
    --original_dev_filepath "./data/spider-dk/Spider-DK.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/spider_dk_tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --num_workers $num_workers \
    --use_parallel
