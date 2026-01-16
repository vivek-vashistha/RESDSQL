#!/usr/bin/env python3
"""
Extract golden SQL queries from Spider2 evaluation suite and combine them
into a single file matching the order of spider2-lite.jsonl
"""

import json
import os
import argparse
from pathlib import Path


def extract_gold_sql(
    spider2_jsonl_path: str,
    sql_dir: str,
    output_path: str
):
    """
    Extract golden SQL queries from individual .sql files and combine them
    in the same order as spider2-lite.jsonl
    
    Args:
        spider2_jsonl_path: Path to spider2-lite.jsonl file
        sql_dir: Path to directory containing .sql files (evaluation_suite/gold/sql/)
        output_path: Path to save the combined SQL file (one SQL per line)
    """
    # Read spider2-lite.jsonl to get instance_ids in order
    instance_ids = []
    with open(spider2_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                instance_id = example.get('instance_id')
                if instance_id:
                    instance_ids.append(instance_id)
    
    print(f"Found {len(instance_ids)} examples in {spider2_jsonl_path}")
    
    # Extract SQL queries in the same order
    gold_sqls = []
    missing_files = []
    
    sql_dir_path = Path(sql_dir)
    
    for instance_id in instance_ids:
        sql_file = sql_dir_path / f"{instance_id}.sql"
        
        if sql_file.exists():
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_query = f.read().strip()
                gold_sqls.append(sql_query)
        else:
            print(f"Warning: SQL file not found for instance_id: {instance_id}")
            missing_files.append(instance_id)
            gold_sqls.append("")  # Empty string for missing SQL
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} SQL files are missing:")
        print(f"  First 10 missing: {missing_files[:10]}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Write to output file (one SQL per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sql in gold_sqls:
            # Replace newlines with spaces and write
            # This ensures one SQL query per line
            sql_single_line = sql.replace('\n', ' ').strip()
            f.write(sql_single_line + '\n')
    
    print(f"\nExtracted {len(gold_sqls)} SQL queries")
    print(f"Saved to: {output_path}")
    print(f"  - Valid SQL queries: {len(gold_sqls) - len(missing_files)}")
    print(f"  - Missing SQL queries: {len(missing_files)}")
    
    return gold_sqls


def main():
    parser = argparse.ArgumentParser(
        description="Extract golden SQL queries from Spider2 evaluation suite"
    )
    parser.add_argument(
        "--spider2_jsonl",
        type=str,
        required=True,
        help="Path to spider2-lite.jsonl file"
    )
    parser.add_argument(
        "--sql_dir",
        type=str,
        required=True,
        help="Path to directory containing .sql files (evaluation_suite/gold/sql/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the combined SQL file (one SQL per line)"
    )
    
    args = parser.parse_args()
    
    extract_gold_sql(
        spider2_jsonl_path=args.spider2_jsonl,
        sql_dir=args.sql_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
