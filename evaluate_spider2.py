"""
Evaluation script for Spider2 dataset.
Executes predicted SQL queries on appropriate databases and compares with gold SQL.
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from utils.database_adapters import get_database_adapter
from utils.spider2_converter import (
    load_spider2_jsonl,
    extract_db_type_from_spider2,
    get_connection_info_from_spider2
)


def execute_query_safely(adapter, sql: str, timeout: int = 30) -> Optional[List]:
    """Execute SQL query with error handling.
    
    Args:
        adapter: Database adapter instance
        sql: SQL query string
        timeout: Query timeout in seconds
        
    Returns:
        Query results or None if execution failed
    """
    try:
        results = adapter.execute_query(sql)
        return results
    except Exception as e:
        print(f"Query execution error: {e}")
        return None


def compare_results(pred_results: Optional[List], gold_results: Optional[List]) -> bool:
    """Compare prediction and gold query results.
    
    Args:
        pred_results: Results from predicted query
        gold_results: Results from gold query
        
    Returns:
        True if results match, False otherwise
    """
    if pred_results is None or gold_results is None:
        return False
    
    # Convert to comparable format (normalize types, sort)
    def normalize_result(result):
        if isinstance(result, (list, tuple)):
            return tuple(sorted([normalize_result(item) for item in result]))
        elif isinstance(result, (int, float)):
            return float(result)
        else:
            return str(result).lower().strip()
    
    try:
        pred_normalized = normalize_result(pred_results)
        gold_normalized = normalize_result(gold_results)
        return pred_normalized == gold_normalized
    except Exception:
        return False


def evaluate_spider2(
    predictions_file: str,
    gold_file: Optional[str] = None,
    spider2_data_file: Optional[str] = None,
    db_mapping_file: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate Spider2 predictions.
    
    Args:
        predictions_file: Path to file with predicted SQL (one per line)
        gold_file: Optional path to file with gold SQL (one per line)
        spider2_data_file: Optional path to Spider2 JSONL file (for database info)
        db_mapping_file: Optional path to database mapping JSON file
        output_file: Optional path to save detailed results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load predictions
    print("Loading predictions...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f if line.strip()]
    
    # Load gold SQL if available
    gold_sqls = None
    if gold_file and os.path.exists(gold_file):
        print("Loading gold SQL...")
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_sqls = [line.strip() for line in f if line.strip()]
    
    # Load Spider2 data for database information
    spider2_data = None
    if spider2_data_file and os.path.exists(spider2_data_file):
        print("Loading Spider2 data...")
        spider2_data = load_spider2_jsonl(spider2_data_file)
    
    # Load database mapping if available
    db_mapping = {}
    if db_mapping_file and os.path.exists(db_mapping_file):
        with open(db_mapping_file, 'r') as f:
            db_mapping = json.load(f)
    
    # Ensure we have enough data
    num_examples = len(predictions)
    if gold_sqls and len(gold_sqls) != num_examples:
        print(f"Warning: Number of predictions ({num_examples}) != number of gold SQLs ({len(gold_sqls)})")
    
    if spider2_data and len(spider2_data) != num_examples:
        print(f"Warning: Number of predictions ({num_examples}) != number of Spider2 examples ({len(spider2_data)})")
    
    # Evaluation metrics
    total = num_examples
    exec_correct = 0
    exact_match = 0
    execution_errors = 0
    
    detailed_results = []
    
    # Cache database adapters
    adapter_cache = {}
    
    print(f"\nEvaluating {total} predictions...")
    for idx in tqdm(range(total)):
        pred_sql = predictions[idx]
        gold_sql = gold_sqls[idx] if gold_sqls else None
        
        # Get database information
        db_id = None
        db_type = 'sqlite'
        connection_info = {}
        
        if spider2_data and idx < len(spider2_data):
            example = spider2_data[idx]
            # Spider2 uses 'db' field as the database identifier
            db_id = example.get('db') or example.get('db_id') or example.get('database_id')
            db_type = extract_db_type_from_spider2(example)
            connection_info = get_connection_info_from_spider2(example, db_type, db_id)
        elif db_mapping:
            # Try to get from mapping (assuming db_id is in mapping)
            # This is a simplified approach - may need adjustment based on actual format
            pass
        
        # Get or create adapter
        adapter_key = f"{db_type}_{db_id}"
        if adapter_key not in adapter_cache:
            try:
                adapter = get_database_adapter(db_type)
                adapter.connect(connection_info)
                adapter_cache[adapter_key] = adapter
            except Exception as e:
                print(f"Warning: Failed to connect to database {db_id} ({db_type}): {e}")
                adapter_cache[adapter_key] = None
        
        adapter = adapter_cache.get(adapter_key)
        
        result = {
            "index": idx,
            "db_id": db_id,
            "db_type": db_type,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "exec_success": False,
            "exact_match": False,
            "error": None
        }
        
        if adapter is None:
            result["error"] = "Database connection failed"
            detailed_results.append(result)
            execution_errors += 1
            continue
        
        # Execute predicted query
        pred_results = execute_query_safely(adapter, pred_sql)
        
        if pred_results is not None:
            result["exec_success"] = True
            exec_correct += 1
            
            # Compare with gold if available
            if gold_sql:
                gold_results = execute_query_safely(adapter, gold_sql)
                if gold_results is not None:
                    if compare_results(pred_results, gold_results):
                        result["exact_match"] = True
                        exact_match += 1
        else:
            result["error"] = "Query execution failed"
            execution_errors += 1
        
        detailed_results.append(result)
    
    # Close all adapters
    for adapter in adapter_cache.values():
        if adapter:
            try:
                adapter.close()
            except:
                pass
    
    # Calculate metrics
    exec_accuracy = exec_correct / total if total > 0 else 0.0
    exact_match_accuracy = exact_match / total if total > 0 else 0.0
    
    metrics = {
        "total": total,
        "exec_correct": exec_correct,
        "exec_accuracy": exec_accuracy,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match_accuracy,
        "execution_errors": execution_errors
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
    parser = argparse.ArgumentParser(description="Evaluate Spider2 predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions file (one SQL per line)")
    parser.add_argument("--gold", type=str, default=None,
                        help="Path to gold SQL file (optional, one SQL per line)")
    parser.add_argument("--spider2_data", type=str, default=None,
                        help="Path to Spider2 JSONL file (for database info)")
    parser.add_argument("--db_mapping", type=str, default=None,
                        help="Path to database mapping JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save detailed evaluation results")
    
    args = parser.parse_args()
    
    metrics = evaluate_spider2(
        predictions_file=args.predictions,
        gold_file=args.gold,
        spider2_data_file=args.spider2_data,
        db_mapping_file=args.db_mapping,
        output_file=args.output
    )
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total examples: {metrics['total']}")
    print(f"Execution correct: {metrics['exec_correct']}")
    print(f"Execution accuracy: {metrics['exec_accuracy']:.4f} ({metrics['exec_accuracy']*100:.2f}%)")
    if metrics['exact_match'] > 0:
        print(f"Exact match: {metrics['exact_match']}")
        print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Execution errors: {metrics['execution_errors']}")
    print("="*50)


if __name__ == "__main__":
    main()
