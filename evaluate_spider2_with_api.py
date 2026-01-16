#!/usr/bin/env python3
"""
Evaluate Spider2 questions using FastAPI to generate SQL, then compare execution results
with golden results from Spider2 repository.

This script:
1. Reads questions from spider2-lite.jsonl
2. Generates SQL using FastAPI server
3. Executes SQL queries on SQLite databases
4. Compares results with golden CSV files from Spider2 repo
"""

import json
import os
import argparse
import requests
import sqlite3
import pandas as pd
import math
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import re


def load_spider2_jsonl(jsonl_path: str) -> List[Dict]:
    """Load Spider2 JSONL file."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_eval_metadata(eval_jsonl_path: str) -> Dict[str, Dict]:
    """Load evaluation metadata (condition_cols, ignore_order, etc.)."""
    metadata = {}
    with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                metadata[item['instance_id']] = item
    return metadata


def generate_sql_via_api(
    question: str,
    db_id: str,
    api_url: str = "http://localhost:8000/infer",
    tables_path: str = "./data/spider2/converted/spider2-lite_tables.json",
    target_type: str = "natsql",
    timeout: int = 300,
    max_retries: int = 2
) -> Optional[str]:
    """Generate SQL using FastAPI server with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                json={
                    "question": question,
                    "db_id": db_id,
                    "target_type": target_type,
                    "spider2_mode": True,
                    "tables_path": tables_path
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('sql')
            else:
                if attempt < max_retries:
                    print(f"API error {response.status_code}, retrying... ({attempt + 1}/{max_retries})")
                    continue
                print(f"API error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying with longer timeout...")
                timeout = timeout * 2  # Double timeout on retry
                continue
            print(f"API request timed out after {timeout}s: {e}")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"API request failed (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                continue
            print(f"API request failed: {e}")
            return None
    
    return None


def execute_sqlite_query(db_path: str, sql: str) -> Optional[pd.DataFrame]:
    """Execute SQL query on SQLite database and return results as DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        memory_conn = sqlite3.connect(':memory:')
        conn.backup(memory_conn)
        
        try:
            df = pd.read_sql_query(sql, memory_conn)
            return df
        finally:
            memory_conn.close()
            conn.close()
    except Exception as e:
        print(f"Query execution error: {e}")
        return None


def normalize_value(value):
    """Normalize a value for comparison."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return str(value).lower().strip()


def compare_dataframes(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    condition_cols: Optional[List[int]] = None,
    ignore_order: bool = False,
    tolerance: float = 1e-2
) -> bool:
    """Compare prediction and gold DataFrames.
    
    Args:
        pred_df: Predicted results DataFrame
        gold_df: Gold results DataFrame
        condition_cols: Column indices to use for comparison (None = all columns)
        ignore_order: Whether to ignore row order
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if results match, False otherwise
    """
    if pred_df is None or gold_df is None:
        return False
    
    # Handle condition_cols (subset of columns to compare)
    if condition_cols:
        if not isinstance(condition_cols, list):
            condition_cols = [condition_cols]
        try:
            gold_df = gold_df.iloc[:, condition_cols]
        except IndexError:
            # If condition_cols indices are invalid, use all columns
            pass
    
    # Normalize DataFrames
    def normalize_df(df):
        return df.applymap(normalize_value)
    
    pred_normalized = normalize_df(pred_df)
    gold_normalized = normalize_df(gold_df)
    
    # Compare column structure
    if len(pred_normalized.columns) != len(gold_normalized.columns):
        return False
    
    # Compare data
    if ignore_order:
        # Sort rows for comparison
        pred_sorted = pred_normalized.sort_values(by=list(pred_normalized.columns)).reset_index(drop=True)
        gold_sorted = gold_normalized.sort_values(by=list(gold_normalized.columns)).reset_index(drop=True)
        
        if len(pred_sorted) != len(gold_sorted):
            return False
        
        # Compare row by row
        for i in range(len(pred_sorted)):
            pred_row = pred_sorted.iloc[i]
            gold_row = gold_sorted.iloc[i]
            
            for col in pred_sorted.columns:
                pred_val = pred_row[col]
                gold_val = gold_row[col]
                
                if pred_val is None and gold_val is None:
                    continue
                if pred_val is None or gold_val is None:
                    return False
                
                if isinstance(pred_val, float) and isinstance(gold_val, float):
                    if not math.isclose(pred_val, gold_val, abs_tol=tolerance):
                        return False
                elif pred_val != gold_val:
                    return False
    else:
        # Exact order comparison
        if len(pred_normalized) != len(gold_normalized):
            return False
        
        # Compare using pandas equals (handles NaN properly)
        try:
            return pred_normalized.equals(gold_normalized)
        except:
            # Fallback to manual comparison
            for i in range(len(pred_normalized)):
                for col in pred_normalized.columns:
                    pred_val = pred_normalized.iloc[i][col]
                    gold_val = gold_normalized.iloc[i][col]
                    
                    if pd.isna(pred_val) and pd.isna(gold_val):
                        continue
                    if pd.isna(pred_val) or pd.isna(gold_val):
                        return False
                    
                    if isinstance(pred_val, float) and isinstance(gold_val, float):
                        if not math.isclose(pred_val, gold_val, abs_tol=tolerance):
                            return False
                    elif pred_val != gold_val:
                        return False
    
    return True


def resolve_gold_csv_path(instance_id: str, gold_result_dir: str) -> Optional[Path]:
    """Resolve path to golden CSV file for an instance_id."""
    # Try exact match first
    base_path = Path(gold_result_dir) / f"{instance_id}.csv"
    if base_path.exists():
        return base_path
    
    # Try pattern matching (for variants like instance_id_a.csv)
    if "_" in instance_id:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    else:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    
    if os.path.exists(gold_result_dir):
        for file in os.listdir(gold_result_dir):
            if pattern.match(file):
                return Path(gold_result_dir) / file
    
    return None


def evaluate_spider2_with_api(
    spider2_jsonl_path: str,
    gold_dir: str,
    api_url: str = "http://localhost:8000/infer",
    tables_path: str = "./data/spider2/converted/spider2-lite_tables.json",
    database_dir: str = "./database",
    output_file: Optional[str] = None,
    filter_sqlite_only: bool = True,
    target_type: str = "natsql",
    api_timeout: int = 300,
    max_retries: int = 2,
    skip_on_error: bool = True
) -> Dict[str, Any]:
    """Evaluate Spider2 questions using FastAPI to generate SQL.
    
    Args:
        spider2_jsonl_path: Path to spider2-lite.jsonl
        gold_dir: Path to Spider2 gold directory (contains exec_result/ and spider2lite_eval.jsonl)
        api_url: FastAPI server URL
        tables_path: Path to tables.json file
        database_dir: Directory containing SQLite databases
        output_file: Optional path to save detailed results
        filter_sqlite_only: Only evaluate SQLite databases (local* instance_ids)
        target_type: Target SQL type ("sql" or "natsql")
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load Spider2 data
    print("Loading Spider2 data...")
    spider2_data = load_spider2_jsonl(spider2_jsonl_path)
    
    # Load evaluation metadata
    eval_jsonl_path = os.path.join(gold_dir, "spider2lite_eval.jsonl")
    if not os.path.exists(eval_jsonl_path):
        print(f"Warning: Evaluation metadata not found at {eval_jsonl_path}")
        eval_metadata = {}
    else:
        eval_metadata = load_eval_metadata(eval_jsonl_path)
    
    # Load golden results directory
    gold_result_dir = os.path.join(gold_dir, "exec_result")
    if not os.path.exists(gold_result_dir):
        raise FileNotFoundError(f"Gold results directory not found: {gold_result_dir}")
    
    # Filter SQLite-only examples if requested
    if filter_sqlite_only:
        sqlite_data = []
        for example in spider2_data:
            instance_id = example.get('instance_id', '')
            # SQLite databases have instance_ids starting with "local"
            if instance_id.startswith('local'):
                sqlite_data.append(example)
        spider2_data = sqlite_data
        print(f"Filtered to {len(spider2_data)} SQLite examples")
    
    # Check if API server is running
    try:
        response = requests.get(f"{api_url.replace('/infer', '/docs')}", timeout=2)
        print("✓ FastAPI server is running")
    except requests.exceptions.RequestException:
        print("✗ FastAPI server is not running!")
        print(f"  Please start the server first: bash start_api.sh")
        raise
    
    # Evaluation metrics
    total = len(spider2_data)
    exec_success = 0
    exec_correct = 0
    errors = []
    
    detailed_results = []
    
    print(f"\nEvaluating {total} questions...")
    print("=" * 60)
    
    for idx, example in enumerate(tqdm(spider2_data, desc="Evaluating")):
        instance_id = example.get('instance_id', '')
        question = example.get('question', '')
        db_id = example.get('db', '')
        
        if not db_id:
            errors.append({
                "instance_id": instance_id,
                "error": "Missing db_id"
            })
            continue
        
        result = {
            "instance_id": instance_id,
            "db_id": db_id,
            "question": question,
            "pred_sql": None,
            "exec_success": False,
            "exec_correct": False,
            "error": None
        }
        
        # Step 1: Generate SQL via API
        pred_sql = generate_sql_via_api(
            question=question,
            db_id=db_id,
            api_url=api_url,
            tables_path=tables_path,
            target_type=target_type,
            timeout=api_timeout,
            max_retries=max_retries
        )
        
        if not pred_sql:
            result["error"] = "Failed to generate SQL via API (timeout or error)"
            detailed_results.append(result)
            errors.append(result)
            if skip_on_error:
                print(f"  ⚠ Skipping {instance_id} due to API error")
                continue
            else:
                # Still try to continue with empty SQL (will fail execution)
                pass
        
        result["pred_sql"] = pred_sql
        
        # Step 2: Execute SQL query
        db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            result["error"] = f"Database file not found: {db_path}"
            detailed_results.append(result)
            errors.append(result)
            continue
        
        pred_df = execute_sqlite_query(db_path, pred_sql)
        
        if pred_df is None:
            result["error"] = "Query execution failed"
            detailed_results.append(result)
            errors.append(result)
            continue
        
        result["exec_success"] = True
        exec_success += 1
        
        # Step 3: Load golden results
        gold_csv_path = resolve_gold_csv_path(instance_id, gold_result_dir)
        
        if not gold_csv_path or not gold_csv_path.exists():
            result["error"] = f"Gold CSV not found for {instance_id}"
            detailed_results.append(result)
            errors.append(result)
            continue
        
        try:
            gold_df = pd.read_csv(gold_csv_path)
        except Exception as e:
            result["error"] = f"Failed to load gold CSV: {e}"
            detailed_results.append(result)
            errors.append(result)
            continue
        
        # Step 4: Compare results
        eval_info = eval_metadata.get(instance_id, {})
        condition_cols = eval_info.get('condition_cols')
        ignore_order = eval_info.get('ignore_order', False)
        
        is_correct = compare_dataframes(
            pred_df=pred_df,
            gold_df=gold_df,
            condition_cols=condition_cols,
            ignore_order=ignore_order
        )
        
        result["exec_correct"] = is_correct
        if is_correct:
            exec_correct += 1
        
        detailed_results.append(result)
    
    # Calculate metrics
    exec_accuracy = exec_success / total if total > 0 else 0.0
    correct_accuracy = exec_correct / total if total > 0 else 0.0
    
    metrics = {
        "total": total,
        "exec_success": exec_success,
        "exec_accuracy": exec_accuracy,
        "exec_correct": exec_correct,
        "correct_accuracy": correct_accuracy,
        "errors": len(errors)
    }
    
    # Save detailed results if requested
    if output_file:
        output_data = {
            "metrics": metrics,
            "results": detailed_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spider2 questions using FastAPI to generate SQL"
    )
    parser.add_argument(
        "--spider2_jsonl",
        type=str,
        required=True,
        help="Path to spider2-lite.jsonl file"
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        required=True,
        help="Path to Spider2 gold directory (contains exec_result/ and spider2lite_eval.jsonl)"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/infer",
        help="FastAPI server URL"
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        default="./data/spider2/converted/spider2-lite_tables.json",
        help="Path to tables.json file"
    )
    parser.add_argument(
        "--database_dir",
        type=str,
        default="./database",
        help="Directory containing SQLite databases"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed evaluation results (JSON)"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="natsql",
        choices=["sql", "natsql"],
        help="Target SQL type"
    )
    parser.add_argument(
        "--all_databases",
        action="store_true",
        help="Evaluate all databases (not just SQLite)"
    )
    parser.add_argument(
        "--api_timeout",
        type=int,
        default=300,
        help="API request timeout in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for failed API requests (default: 2)"
    )
    parser.add_argument(
        "--fail_on_error",
        action="store_true",
        help="Stop evaluation on first error (default: skip and continue)"
    )
    
    args = parser.parse_args()
    
    metrics = evaluate_spider2_with_api(
        spider2_jsonl_path=args.spider2_jsonl,
        gold_dir=args.gold_dir,
        api_url=args.api_url,
        tables_path=args.tables_path,
        database_dir=args.database_dir,
        output_file=args.output,
        filter_sqlite_only=not args.all_databases,
        target_type=args.target_type,
        api_timeout=args.api_timeout,
        max_retries=args.max_retries,
        skip_on_error=not args.fail_on_error
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total examples: {metrics['total']}")
    print(f"Execution success: {metrics['exec_success']}")
    print(f"Execution success rate: {metrics['exec_accuracy']:.4f} ({metrics['exec_accuracy']*100:.2f}%)")
    print(f"Execution correct: {metrics['exec_correct']}")
    print(f"Correct accuracy: {metrics['correct_accuracy']:.4f} ({metrics['correct_accuracy']*100:.2f}%)")
    print(f"Errors: {metrics['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
