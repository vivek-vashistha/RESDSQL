#!/usr/bin/env bash

# Inference script for Spider2 dataset
# Usage: sh scripts/inference/infer_text2natsql_spider2.sh <model_scale> <spider2_type> [input_jsonl_path]
# Example: sh scripts/inference/infer_text2natsql_spider2.sh base lite

# set -e

# Use conda Python if available, otherwise use system Python
if [ -f "/opt/anaconda3/envs/resdsql/bin/python" ]; then
    PYTHON="/opt/anaconda3/envs/resdsql/bin/python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python"
fi

# Set device to CPU for Mac (change to "mps" for Apple Silicon GPU, or "0" for CUDA GPU)
device="cpu"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql_spider2.json"

# Model configuration
if [ $1 = "base" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-base/checkpoint-14352"
    text2natsql_model_bs=16
elif [ $1 = "large" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-large/checkpoint-21216"
    text2natsql_model_bs=8
elif [ $1 = "3b" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-78302"
    text2natsql_model_bs=6
else
    echo "The first arg must be in [base, large, 3b]."
    exit 1
fi

model_name="resdsql_$1_natsql"

# Spider2 dataset configuration
if [ $2 = "lite" ]
then
    # Spider2-Lite (BigQuery, Snowflake, SQLite)
    input_jsonl_path=${3:-"./data/spider2/spider2-lite.jsonl"}
    dataset_name="spider2-lite"
    output_dir="./data/spider2/converted"
    db_path="./database"
elif [ $2 = "snow" ]
then
    # Spider2-Snow (Snowflake only)
    input_jsonl_path=${3:-"./data/spider2/spider2-snow.jsonl"}
    dataset_name="spider2-snow"
    output_dir="./data/spider2/converted"
    db_path="./database"
else
    echo "The second arg must be in [lite, snow]."
    exit 1
fi

# Output paths
table_path="${output_dir}/${dataset_name}_tables.json"
input_dataset_path="${output_dir}/${dataset_name}_dataset.json"
db_mapping_path="${output_dir}/${dataset_name}_db_mapping.json"
output="./predictions/${dataset_name}/$model_name/pred.sql"

# Create output directories
mkdir -p "${output_dir}"
mkdir -p "$(dirname "$output")"

echo "=========================================="
echo "Spider2 Inference Pipeline"
echo "=========================================="
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "Input: $input_jsonl_path"
echo "Output: $output"
echo "=========================================="

# Step 1: Convert Spider2 JSONL to RESDSQL format
echo ""
echo "Step 1: Converting Spider2 JSONL to RESDSQL format..."
if [ ! -f "$table_path" ] || [ ! -f "$input_dataset_path" ]; then
    $PYTHON utils/spider2_converter.py \
        --input "$input_jsonl_path" \
        --output_dir "$output_dir" \
        --dataset_name "$dataset_name"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert Spider2 data"
        exit 1
    fi
else
    # Check if existing files have valid schemas (at least one table)
    echo "  Checking existing converted files..."
    invalid_schemas=$($PYTHON -c "
import json
import sys
try:
    with open('$table_path', 'r') as f:
        tables = json.load(f)
    empty = sum(1 for t in tables if not t.get('table_names_original') or len(t.get('table_names_original', [])) == 0)
    if empty > 0:
        print(f'{empty}')
        sys.exit(1)
    else:
        print('0')
        sys.exit(0)
except Exception as e:
    print(f'Error checking: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ "$invalid_schemas" != "0" ]; then
        echo "  Warning: Found $invalid_schemas empty schemas in existing files"
        echo "  Regenerating converted files..."
        rm -f "$table_path" "$input_dataset_path"
        $PYTHON utils/spider2_converter.py \
            --input "$input_jsonl_path" \
            --output_dir "$output_dir" \
            --dataset_name "$dataset_name"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to convert Spider2 data"
            exit 1
        fi
    else
        echo "  Using existing converted files (all schemas valid)"
    fi
fi

# Step 2: Prepare table file for NatSQL
echo ""
echo "Step 2: Preparing NatSQL table file..."
$PYTHON NatSQL/table_transform.py \
    --in_file "$table_path" \
    --out_file "$tables_for_natsql" \
    --correct_col_type \
    --remove_start_table \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path "$db_path"

if [ $? -ne 0 ]; then
    echo "Warning: Table transformation may have failed, continuing..."
fi

# Step 3: Preprocess test set
echo ""
echo "Step 3: Preprocessing test set..."
preprocessed_output="./data/preprocessed_data/preprocessed_test_natsql_spider2.json"
if [ -f "$preprocessed_output" ]; then
    echo "  Preprocessed file already exists, skipping preprocessing..."
    echo "  File: $preprocessed_output"
else
    $PYTHON preprocessing.py \
        --mode "test" \
        --table_path "$table_path" \
        --input_dataset_path "$input_dataset_path" \
        --output_dataset_path "$preprocessed_output" \
        --db_path "$db_path" \
        --target_type "natsql" \
        --spider2_mode

    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed"
        exit 1
    fi
fi

# Step 4: Predict probability for each schema item
echo ""
echo "Step 4: Classifying schema items..."
schema_classifier_output="./data/preprocessed_data/test_with_probs_natsql_spider2.json"
if [ -f "$schema_classifier_output" ]; then
    echo "  Schema classification file already exists, skipping..."
    echo "  File: $schema_classifier_output"
else
    $PYTHON schema_item_classifier.py \
        --batch_size 32 \
        --device "$device" \
        --seed 42 \
        --save_path "./models/text2natsql_schema_item_classifier" \
        --dev_filepath "./data/preprocessed_data/preprocessed_test_natsql_spider2.json" \
        --output_filepath "$schema_classifier_output" \
        --use_contents \
        --mode "test"

    if [ $? -ne 0 ]; then
        echo "Error: Schema classification failed"
        exit 1
    fi
fi

# Step 5: Generate text2natsql test set
echo ""
echo "Step 5: Generating text2natsql test set..."
ranked_dataset_output="./data/preprocessed_data/resdsql_test_natsql_spider2.json"
if [ -f "$ranked_dataset_output" ]; then
    echo "  Ranked dataset file already exists, skipping..."
    echo "  File: $ranked_dataset_output"
else
    $PYTHON text2sql_data_generator.py \
        --input_dataset_path "./data/preprocessed_data/test_with_probs_natsql_spider2.json" \
        --output_dataset_path "$ranked_dataset_output" \
        --topk_table_num 4 \
        --topk_column_num 5 \
        --mode "test" \
        --use_contents \
        --output_skeleton \
        --target_type "natsql"

    if [ $? -ne 0 ]; then
        echo "Error: Dataset generation failed"
        exit 1
    fi
fi

# Step 6: Inference using the text2natsql model
echo ""
echo "Step 6: Running inference..."
$PYTHON text2sql.py \
    --batch_size $text2natsql_model_bs \
    --device "$device" \
    --seed 42 \
    --save_path "$text2natsql_model_save_path" \
    --mode "test" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql_spider2.json" \
    --original_dev_filepath "$input_dataset_path" \
    --db_path "$db_path" \
    --tables_for_natsql "$tables_for_natsql" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output "$output"

if [ $? -ne 0 ]; then
    echo "Error: Inference failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Inference Complete!"
echo "=========================================="
echo "Predictions saved to: $output"
echo "=========================================="
