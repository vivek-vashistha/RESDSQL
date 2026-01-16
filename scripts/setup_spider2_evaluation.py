#!/usr/bin/env python3
"""
Setup script to check and copy necessary files from Spider2 repository for evaluation.
"""

import os
import shutil
import argparse
from pathlib import Path


def check_predictions_file(predictions_path: str) -> bool:
    """Check if predictions file exists."""
    if os.path.exists(predictions_path):
        print(f"✓ Predictions file exists: {predictions_path}")
        return True
    else:
        print(f"✗ Predictions file missing: {predictions_path}")
        return False


def copy_spider2_evaluation_script(
    spider2_repo_path: str,
    target_dir: str = "scripts/evaluation"
):
    """Copy official Spider2 evaluation script if needed."""
    source_eval = Path(spider2_repo_path) / "spider2-lite" / "evaluation_suite" / "evaluate.py"
    source_eval_utils = Path(spider2_repo_path) / "spider2-lite" / "evaluation_suite" / "evaluate_utils.py"
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    if source_eval.exists():
        target_eval = target_dir_path / "spider2_official_evaluate.py"
        if not target_eval.exists():
            shutil.copy2(source_eval, target_eval)
            print(f"✓ Copied: {source_eval} -> {target_eval}")
            copied_files.append(str(target_eval))
        else:
            print(f"⚠ Already exists: {target_eval}")
    else:
        print(f"✗ Source file not found: {source_eval}")
    
    if source_eval_utils.exists():
        target_eval_utils = target_dir_path / "spider2_evaluate_utils.py"
        if not target_eval_utils.exists():
            shutil.copy2(source_eval_utils, target_eval_utils)
            print(f"✓ Copied: {source_eval_utils} -> {target_eval_utils}")
            copied_files.append(str(target_eval_utils))
        else:
            print(f"⚠ Already exists: {target_eval_utils}")
    else:
        print(f"✗ Source file not found: {source_eval_utils}")
    
    return copied_files


def check_gold_sql_file(gold_sql_path: str) -> bool:
    """Check if golden SQL file exists."""
    if os.path.exists(gold_sql_path):
        print(f"✓ Golden SQL file exists: {gold_sql_path}")
        # Check if it has content
        with open(gold_sql_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            non_empty = sum(1 for l in lines if l)
            print(f"  - Contains {non_empty} non-empty SQL queries out of {len(lines)} total lines")
        return True
    else:
        print(f"✗ Golden SQL file missing: {gold_sql_path}")
        return False


def check_spider2_data_file(spider2_jsonl_path: str) -> bool:
    """Check if Spider2 data file exists."""
    if os.path.exists(spider2_jsonl_path):
        print(f"✓ Spider2 data file exists: {spider2_jsonl_path}")
        with open(spider2_jsonl_path, 'r') as f:
            lines = sum(1 for _ in f)
            print(f"  - Contains {lines} examples")
        return True
    else:
        print(f"✗ Spider2 data file missing: {spider2_jsonl_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup and check Spider2 evaluation files"
    )
    parser.add_argument(
        "--spider2_repo",
        type=str,
        default="/Users/vivekvashistha/Projects/Clients/Turing/Projects/Spider2",
        help="Path to Spider2 repository"
    )
    parser.add_argument(
        "--copy_eval_script",
        action="store_true",
        help="Copy official Spider2 evaluation script"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="predictions/spider2-lite/resdsql_base_natsql/pred.sql",
        help="Path to predictions file"
    )
    parser.add_argument(
        "--gold_sql",
        type=str,
        default="data/spider2/spider2-lite-gold.sql",
        help="Path to golden SQL file"
    )
    parser.add_argument(
        "--spider2_data",
        type=str,
        default="data/spider2/spider2-lite.jsonl",
        help="Path to Spider2 data file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Spider2 Evaluation Setup Check")
    print("=" * 60)
    print()
    
    # Check required files
    print("Checking required files...")
    print("-" * 60)
    
    has_predictions = check_predictions_file(args.predictions)
    has_gold_sql = check_gold_sql_file(args.gold_sql)
    has_spider2_data = check_spider2_data_file(args.spider2_data)
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if not has_predictions:
        print("\n⚠️  MISSING: Predictions file")
        print("   You need to run inference first:")
        print(f"   sh scripts/inference/infer_text2natsql_spider2.sh base lite")
        print()
    
    if not has_gold_sql:
        print("\n⚠️  MISSING: Golden SQL file")
        print("   Run the extraction script:")
        print(f"   python scripts/extract_spider2_gold_sql.py \\")
        print(f"       --spider2_jsonl {args.spider2_data} \\")
        print(f"       --sql_dir {args.spider2_repo}/spider2-lite/evaluation_suite/gold/sql \\")
        print(f"       --output {args.gold_sql}")
        print()
    
    if not has_spider2_data:
        print("\n⚠️  MISSING: Spider2 data file")
        print(f"   Expected at: {args.spider2_data}")
        print()
    
    # Copy evaluation script if requested
    if args.copy_eval_script:
        print()
        print("=" * 60)
        print("Copying Spider2 Evaluation Scripts")
        print("=" * 60)
        copied = copy_spider2_evaluation_script(args.spider2_repo)
        if copied:
            print(f"\n✓ Copied {len(copied)} file(s)")
            print("\nTo use the official Spider2 evaluation script:")
            print("  python scripts/evaluation/spider2_official_evaluate.py \\")
            print("      --result_dir predictions/spider2-lite/resdsql_base_natsql \\")
            print("      --mode sql")
        else:
            print("\n⚠ No files copied (already exist or source not found)")
    
    # Final status
    print()
    print("=" * 60)
    if has_predictions and has_gold_sql and has_spider2_data:
        print("✓ All required files are present!")
        print("\nYou can now run evaluation:")
        print(f"  python evaluate_spider2.py \\")
        print(f"      --predictions {args.predictions} \\")
        print(f"      --gold {args.gold_sql} \\")
        print(f"      --spider2_data {args.spider2_data} \\")
        print(f"      --output evaluation_results.json")
    else:
        print("⚠ Some files are missing. Please address the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
