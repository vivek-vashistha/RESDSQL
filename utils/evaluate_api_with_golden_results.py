"""
Script to evaluate API-generated SQL queries against golden results.

1. Filters entries with gold_sql_available: true
2. Calls API to generate SQL
3. Executes SQL on SQLite databases
4. Saves results to CSV and JSON
5. Compares with golden CSV files (handling multiple variants)
"""

import json
import os
import sys
import sqlite3
import pandas as pd
import requests
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from functools import lru_cache
import re

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import comparison functions directly to avoid BigQuery dependency
# (We only need SQLite, not BigQuery/Snowflake)
from functools import lru_cache

@lru_cache(maxsize=None)
def load_gold_csv(file_path: str) -> pd.DataFrame:
    """Cache gold CSV loads to avoid repeated disk reads during evaluation."""
    return pd.read_csv(file_path)


def resolve_gold_paths(instance_id: str, gold_result_dir: str):
    """Resolve paths to golden CSV files for an instance_id (handles _a, _b, _c variants)."""
    base_path = Path(gold_result_dir) / f"{instance_id}.csv"
    if base_path.exists():
        return [base_path], True

    if "_" in instance_id:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    else:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")

    csv_files = sorted(
        file for file in os.listdir(gold_result_dir)
        if pattern.match(file)
    )
    return [Path(gold_result_dir) / file for file in csv_files], False


def compare_pandas_table(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols=None, ignore_order: bool = False) -> int:
    """Compare prediction and gold DataFrames. Returns 1 if match, 0 otherwise."""
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]

        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))

        if len(v1) != len(v2):
            return False

        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    pred_cols = pred
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1

    for gold_vector in t_gold_list:
        if not any(vectors_match(gold_vector, pred_vector, ignore_order_=ignore_order) for pred_vector in t_pred_list):
            score = 0
            break

    return score


def compare_multi_pandas_table(pred: pd.DataFrame, multi_gold, multi_condition_cols=None, multi_ignore_order=False) -> int:
    """Compare prediction against multiple golden DataFrames. Returns 1 if matches any, 0 otherwise."""
    if not multi_gold:
        return 0

    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """Load the dataset JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def call_api(
    question: str,
    db_id: str,
    api_url: str = "http://localhost:8000/infer",
    tables_path: str = "./data/spider2/converted/spider2-lite_tables.json",
    target_type: str = "natsql",
    timeout: int = 600,
    max_retries: int = 2,
    num_return_sequences: int = 1,
    num_beams: int = 4
) -> Optional[Dict[str, Any]]:
    """Call the API to generate SQL."""
    payload = {
        "question": question,
        "db_id": db_id,
        "target_type": target_type,
        "spider2_mode": True,
        "tables_path": tables_path,
        "num_return_sequences": num_return_sequences,  # Reduce to speed up
        "num_beams": num_beams  # Reduce to speed up
    }
    
    print(f"  Payload: question length={len(question)}, db_id={db_id}, target_type={target_type}")
    
    for attempt in range(max_retries):
        try:
            # Use a longer timeout for the connection and read separately
            print(f"  Sending request (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(api_url, json=payload, timeout=(30, timeout))
            response.raise_for_status()
            result = response.json()
            
            # Log full API response
            print(f"  API response received:")
            print(f"    Status Code: {response.status_code}")
            print(f"    SQL: {result.get('sql', 'N/A')[:200]}..." if len(result.get('sql', '')) > 200 else f"    SQL: {result.get('sql', 'N/A')}")
            print(f"    SQL Length: {len(result.get('sql', ''))} chars")
            print(f"    Execution Success: {result.get('execution_success', 'N/A')}")
            print(f"    Error: {result.get('error', 'None')}")
            if result.get('input_sequence'):
                print(f"    Input Sequence Length: {len(result.get('input_sequence', ''))} chars")
            print(f"    Full Response: {json.dumps(result, indent=2)}")
            
            return result
        except requests.exceptions.Timeout as e:
            print(f"  === API Call Timeout ===")
            print(f"  Attempt: {attempt + 1}/{max_retries}")
            print(f"  Timeout: {timeout}s")
            print(f"  Error: {str(e)}")
            print(f"  Error Type: {type(e).__name__}")
            if attempt < max_retries - 1:
                print(f"  Retrying with {timeout}s timeout...")
                continue
            else:
                print(f"  API call timed out after {max_retries} attempts")
                print(f"  ======================")
                return None
        except requests.exceptions.HTTPError as e:
            print(f"  === API HTTP Error ===")
            print(f"  Attempt: {attempt + 1}/{max_retries}")
            print(f"  Error: {str(e)}")
            print(f"  Error Type: {type(e).__name__}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response Status Code: {e.response.status_code}")
                print(f"  Response Headers: {dict(e.response.headers)}")
                print(f"  Response Body: {e.response.text[:500]}..." if len(e.response.text) > 500 else f"  Response Body: {e.response.text}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
                continue
            else:
                print(f"  API call failed after {max_retries} attempts")
                print(f"  ====================")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  === API Request Error ===")
            print(f"  Attempt: {attempt + 1}/{max_retries}")
            print(f"  Error: {str(e)}")
            print(f"  Error Type: {type(e).__name__}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
                continue
            else:
                print(f"  API call failed after {max_retries} attempts")
                print(f"  ======================")
                return None
        except Exception as e:
            print(f"  === Unexpected API Error ===")
            print(f"  Attempt: {attempt + 1}/{max_retries}")
            print(f"  Error: {str(e)}")
            print(f"  Error Type: {type(e).__name__}")
            import traceback
            print(f"  Traceback:\n{traceback.format_exc()}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
                continue
            else:
                print(f"  API call failed after {max_retries} attempts")
                print(f"  =========================")
                return None
    
    return None


def execute_sqlite_query(db_path: str, sql: str) -> Optional[pd.DataFrame]:
    """Execute SQL query on SQLite database and return results as DataFrame."""
    try:
        print(f"    Executing SQL: {sql[:200]}..." if len(sql) > 200 else f"    Executing SQL: {sql}")
        
        # Check if database file exists
        if not os.path.exists(db_path):
            print(f"    ERROR: Database file not found: {db_path}")
            return None
        
        conn = sqlite3.connect(db_path)
        memory_conn = sqlite3.connect(':memory:')
        conn.backup(memory_conn)
        
        try:
            df = pd.read_sql_query(sql, memory_conn)
            print(f"    SQL executed successfully. Result shape: {df.shape}, Columns: {list(df.columns)}")
            if df.empty:
                print(f"    WARNING: SQL returned empty result set")
            else:
                print(f"    First row: {df.iloc[0].to_dict()}")
            return df
        except sqlite3.Error as e:
            print(f"    SQLite error: {e}")
            print(f"    SQL that failed: {sql}")
            return None
        except Exception as e:
            print(f"    Query execution error: {e}")
            print(f"    Error type: {type(e).__name__}")
            import traceback
            print(f"    Traceback: {traceback.format_exc()}")
            return None
        finally:
            memory_conn.close()
            conn.close()
    except Exception as e:
        print(f"    Database connection error: {e}")
        import traceback
        print(f"    Traceback: {traceback.format_exc()}")
        return None


def dataframe_to_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dictionaries for JSON serialization."""
    if df is None or df.empty:
        return []
    
    # Convert to dict, handling NaN values
    # Use a simpler approach: iterate through rows and handle NaN manually
    result = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            
            # Check if value is a Series first (shouldn't happen with iterrows, but be safe)
            if isinstance(value, pd.Series):
                value = value.iloc[0] if len(value) > 0 else None
            
            # Check if value is NaN/None
            is_na = False
            try:
                # pd.isna works on scalars
                if value is None:
                    is_na = True
                else:
                    is_na = pd.isna(value)
                    # If pd.isna returns a Series (shouldn't happen), extract scalar
                    if isinstance(is_na, pd.Series):
                        is_na = is_na.iloc[0] if len(is_na) > 0 else False
            except (ValueError, TypeError):
                # If checking fails, assume not NaN
                is_na = False
            
            if is_na:
                row_dict[col] = None
            elif isinstance(value, (int, float)):
                # Check for NaN/Inf using math
                try:
                    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                        row_dict[col] = None
                    else:
                        # Try to convert to int if it's a whole number, otherwise float
                        if isinstance(value, float) and value.is_integer():
                            row_dict[col] = int(value)
                        else:
                            row_dict[col] = float(value) if isinstance(value, float) else int(value)
                except (ValueError, OverflowError, TypeError):
                    row_dict[col] = str(value)
            elif isinstance(value, (str, bool)):
                row_dict[col] = value
            elif isinstance(value, pd.Timestamp):
                # Convert Timestamp to string
                row_dict[col] = str(value)
            else:
                # Convert everything else to string
                try:
                    row_dict[col] = str(value)
                except Exception:
                    row_dict[col] = None
        
        result.append(row_dict)
    
    return result


def load_eval_metadata(eval_jsonl_path: str) -> Dict[str, Dict]:
    """Load evaluation metadata (condition_cols, ignore_order, etc.)."""
    metadata = {}
    if not os.path.exists(eval_jsonl_path):
        return metadata
    
    with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instance_id = item.get('instance_id')
                if instance_id:
                    metadata[instance_id] = {
                        'condition_cols': item.get('condition_cols'),
                        'ignore_order': item.get('ignore_order', False)
                    }
    
    return metadata


def evaluate_api_results(
    input_json_path: str,
    output_json_path: str,
    api_url: str = "http://localhost:8000/infer",
    tables_path: str = "./data/spider2/converted/spider2-lite_tables.json",
    database_dir: str = "./database",
    gold_result_dir: str = "../Spider2/spider2-lite/evaluation_suite/gold/exec_result",
    eval_metadata_path: str = "../Spider2/spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl",
    result_csv_dir: str = "./data/spider2/execution_results",
    target_type: str = "natsql",
    api_timeout: int = 600,
    max_retries: int = 2,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Main evaluation function."""
    
    # Load dataset
    print(f"Loading dataset from {input_json_path}...")
    dataset = load_dataset(input_json_path)
    print(f"Loaded {len(dataset)} entries")
    
    # Filter for entries with gold_sql_available: true
    filtered_entries = [
        entry for entry in dataset 
        if entry.get('metadata', {}).get('gold_sql_available', False) is True
    ]
    print(f"Found {len(filtered_entries)} entries with gold_sql_available: true")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        filtered_entries = filtered_entries[:limit]
        print(f"Limited to first {len(filtered_entries)} entries for testing")
    
    # Load evaluation metadata
    eval_metadata = load_eval_metadata(eval_metadata_path)
    print(f"Loaded evaluation metadata for {len(eval_metadata)} instances")
    
    # Create output directory for CSV files
    os.makedirs(result_csv_dir, exist_ok=True)
    
    # Process each entry
    results = []
    stats = {
        "total": len(filtered_entries),
        "api_success": 0,
        "api_failed": 0,
        "exec_success": 0,
        "exec_failed": 0,
        "gold_found": 0,
        "gold_not_found": 0,
        "comparison_correct": 0,
        "comparison_incorrect": 0
    }
    
    for entry in tqdm(filtered_entries, desc="Processing entries"):
        instance_id = entry.get('metadata', {}).get('instance_id', '')
        db_id = entry.get('metadata', {}).get('database', '')
        question = entry.get('turns', [{}])[0].get('input', '')
        
        result_entry = {
            "id": entry.get('id'),
            "turns": entry.get('turns', []).copy(),
            "metadata": entry.get('metadata', {}).copy(),
            "gold_sql": entry.get('gold_sql', ''),
            "api_response": None,
            "api_sql": None,
            "execution_result": None,
            "execution_result_csv": None,
            "execution_success": False,
            "gold_comparison": {
                "gold_files_found": [],
                "gold_files_compared": [],
                "matches_any_gold": False,
                "comparison_details": None
            }
        }
        
        # Step 1: Call API
        print(f"\n[{instance_id}] Calling API...")
        print(f"  Question: {question[:100]}..." if len(question) > 100 else f"  Question: {question}")
        print(f"  Database: {db_id}")
        
        start_time = time.time()
        
        api_response = call_api(
            question=question,
            db_id=db_id,
            api_url=api_url,
            tables_path=tables_path,
            target_type=target_type,
            timeout=api_timeout,
            max_retries=max_retries,
            num_return_sequences=1,  # Reduce to speed up - only need one valid SQL
            num_beams=4  # Reduce to speed up
        )
        
        elapsed_time = time.time() - start_time
        print(f"  API call took {elapsed_time:.2f} seconds")
        
        if api_response is None:
            print(f"\n  === API Response Details ===")
            print(f"  API Call Status: FAILED")
            print(f"  API Call Duration: {elapsed_time:.2f} seconds")
            print(f"  Error: API call failed or timed out")
            print(f"  ============================\n")
            result_entry["api_response"] = {"error": "API call failed or timed out"}
            result_entry["api_call_duration_seconds"] = elapsed_time
            stats["api_failed"] += 1
            results.append(result_entry)
            continue
        
        stats["api_success"] += 1
        result_entry["api_response"] = api_response
        result_entry["api_call_duration_seconds"] = elapsed_time
        
        # Log complete API response details
        print(f"\n  === API Response Details ===")
        print(f"  API Call Status: SUCCESS")
        print(f"  API Call Duration: {elapsed_time:.2f} seconds")
        print(f"  Full API Response:")
        print(f"    - SQL: {api_response.get('sql', 'N/A')[:200]}..." if len(api_response.get('sql', '')) > 200 else f"    - SQL: {api_response.get('sql', 'N/A')}")
        print(f"    - SQL Length: {len(api_response.get('sql', ''))} characters")
        print(f"    - Execution Success (from API): {api_response.get('execution_success', 'N/A')}")
        print(f"    - Error (from API): {api_response.get('error', 'None')}")
        if api_response.get('input_sequence'):
            print(f"    - Input Sequence Length: {len(api_response.get('input_sequence', ''))} characters")
        print(f"  ============================\n")
        
        # Extract SQL from API response
        api_sql = api_response.get('sql', '')
        execution_success = api_response.get('execution_success', False)
        api_error = api_response.get('error')
        
        if not api_sql:
            result_entry["api_sql"] = None
            result_entry["execution_result"] = {"error": f"No SQL returned from API. API error: {api_error}"}
            results.append(result_entry)
            continue
        
        # Initialize execution result variables
        exec_df = None
        result_entry["api_sql"] = api_sql
        
        # Check if API returned "sql placeholder" (indicates all NatSQL sequences failed to parse)
        if api_sql.strip().lower() == "sql placeholder":
            print(f"  WARNING: API returned 'sql placeholder' - all NatSQL sequences failed to parse")
            print(f"  This means the generated NatSQL contains invalid column/table names")
            result_entry["execution_result"] = {"error": "API returned 'sql placeholder' - all generated NatSQL sequences failed to parse. The generated NatSQL likely contains invalid column/table names that don't exist in the schema."}
            result_entry["execution_result_csv"] = None
            result_entry["execution_success"] = False
            stats["exec_failed"] += 1
            # Skip execution but continue to gold comparison (will mark as incorrect)
        else:
            # Step 2: Execute SQL
            print(f"[{instance_id}] Executing SQL...")
            db_path = os.path.join(database_dir, db_id, f"{db_id}.sqlite")
            print(f"    Database path: {db_path}")
            
            if not os.path.exists(db_path):
                print(f"    ERROR: Database file not found")
                result_entry["execution_result"] = {"error": f"Database not found: {db_path}"}
                result_entry["execution_result_csv"] = None
                result_entry["execution_success"] = False
                stats["exec_failed"] += 1
                # Continue to gold comparison
            else:
                exec_df = execute_sqlite_query(db_path, api_sql)
                
                if exec_df is None:
                    print(f"    ERROR: SQL execution failed - no results returned")
                    result_entry["execution_result"] = {"error": "Query execution failed"}
                    result_entry["execution_result_csv"] = None
                    result_entry["execution_success"] = False
                    stats["exec_failed"] += 1
                    # Continue to gold comparison
                else:
                    stats["exec_success"] += 1
                    result_entry["execution_success"] = True
                    
                    # Convert DataFrame to dict list for JSON
                    result_entry["execution_result"] = dataframe_to_dict_list(exec_df)
                    
                    # Step 3: Save to CSV
                    csv_filename = f"{instance_id}.csv"
                    csv_path = os.path.join(result_csv_dir, csv_filename)
                    exec_df.to_csv(csv_path, index=False)
                    print(f"[{instance_id}] Saved results to {csv_path}")
                    print(f"    Saved {len(exec_df)} rows with columns: {list(exec_df.columns)}")
                    result_entry["execution_result_csv"] = csv_filename
        
        # Step 4: Compare with golden results
        print(f"[{instance_id}] Comparing with golden results...")
        gold_paths, is_single = resolve_gold_paths(instance_id, gold_result_dir)
        
        if not gold_paths:
            print(f"  No golden files found for {instance_id}")
            result_entry["gold_comparison"]["gold_files_found"] = []
            stats["gold_not_found"] += 1
            results.append(result_entry)
            continue
        
        print(f"  Found {len(gold_paths)} golden file(s): {[str(p.name) for p in gold_paths]}")
        stats["gold_found"] += 1
        result_entry["gold_comparison"]["gold_files_found"] = [str(p) for p in gold_paths]
        
        # Get evaluation metadata
        eval_info = eval_metadata.get(instance_id, {})
        condition_cols = eval_info.get('condition_cols')
        ignore_order = eval_info.get('ignore_order', False)
        if condition_cols:
            print(f"  Using condition columns: {condition_cols}")
        if ignore_order:
            print(f"  Ignoring row order in comparison")
        
        # Compare with golden files
        comparison_details = []
        matches_any = False
        
        # Check if we have execution results to compare
        # Note: exec_df might not be defined if execution failed, so check execution_success first
        if not result_entry.get("execution_success", False):
            print(f"  Cannot compare: Execution failed")
            result_entry["gold_comparison"]["matches_any_gold"] = False
            reason = "Execution failed"
            if result_entry.get("api_sql", "").strip().lower() == "sql placeholder":
                reason = "API returned 'sql placeholder' - all NatSQL sequences failed to parse"
            result_entry["gold_comparison"]["comparison_details"] = [{
                "gold_file": str(gold_paths[0]),
                "match": False,
                "reason": reason
            }]
            stats["comparison_incorrect"] += 1
            results.append(result_entry)
            continue
        
        # exec_df should be defined if execution_success is True
        if 'exec_df' not in locals() or exec_df is None:
            print(f"  Cannot compare: No execution results available")
            result_entry["gold_comparison"]["matches_any_gold"] = False
            result_entry["gold_comparison"]["comparison_details"] = [{
                "gold_file": str(gold_paths[0]),
                "match": False,
                "reason": "No execution results available"
            }]
            stats["comparison_incorrect"] += 1
            results.append(result_entry)
            continue
        
        if is_single:
            # Single golden file
            try:
                print(f"  Loading golden file: {gold_paths[0].name}")
                gold_df = load_gold_csv(str(gold_paths[0]))
                print(f"  Golden file shape: {gold_df.shape}, Execution result shape: {exec_df.shape}")
                print(f"  Golden columns: {list(gold_df.columns)}")
                print(f"  Execution columns: {list(exec_df.columns)}")
                print(f"  Golden data (first 3 rows):")
                print(gold_df.head(3).to_string())
                print(f"  Execution data (first 3 rows):")
                print(exec_df.head(3).to_string())
                score = compare_pandas_table(exec_df, gold_df, condition_cols, ignore_order)
                matches_any = (score == 1)
                print(f"  Comparison result: {'MATCH' if matches_any else 'NO MATCH'} (score: {score})")
                if not matches_any:
                    print(f"  Reason: Results do not match - check column names, data types, or row values")
                comparison_details.append({
                    "gold_file": str(gold_paths[0]),
                    "match": matches_any,
                    "score": score,
                    "gold_shape": list(gold_df.shape),
                    "exec_shape": list(exec_df.shape),
                    "gold_columns": list(gold_df.columns),
                    "exec_columns": list(exec_df.columns)
                })
            except Exception as e:
                print(f"  Error during comparison: {str(e)}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                comparison_details.append({
                    "gold_file": str(gold_paths[0]),
                    "error": str(e)
                })
        else:
            # Multiple golden files - check if matches any
            print(f"  Comparing with {len(gold_paths)} golden file variants...")
            gold_dfs = []
            for gold_path in gold_paths:
                try:
                    print(f"    Loading: {gold_path.name}")
                    gold_df = load_gold_csv(str(gold_path))
                    print(f"      Shape: {gold_df.shape}")
                    gold_dfs.append(gold_df)
                    comparison_details.append({
                        "gold_file": str(gold_path),
                        "loaded": True,
                        "shape": list(gold_df.shape)
                    })
                except Exception as e:
                    print(f"      Error loading {gold_path.name}: {str(e)}")
                    comparison_details.append({
                        "gold_file": str(gold_path),
                        "error": str(e)
                    })
            
            if gold_dfs:
                # Use multi-comparison function
                try:
                    print(f"  Execution result shape: {exec_df.shape}")
                    score = compare_multi_pandas_table(
                        exec_df, 
                        gold_dfs, 
                        condition_cols if isinstance(condition_cols, list) and len(condition_cols) > 0 and isinstance(condition_cols[0], list) else [condition_cols] * len(gold_dfs),
                        [ignore_order] * len(gold_dfs)
                    )
                    matches_any = (score == 1)
                    print(f"  Multi-comparison result: {'MATCH' if matches_any else 'NO MATCH'} (score: {score})")
                    comparison_details.append({
                        "multi_comparison": True,
                        "match": matches_any,
                        "score": score
                    })
                except Exception as e:
                    print(f"  Error during multi-comparison: {str(e)}")
                    import traceback
                    print(f"  Traceback: {traceback.format_exc()}")
                    comparison_details.append({
                        "multi_comparison": True,
                        "error": str(e)
                    })
            else:
                print(f"  No golden files could be loaded for comparison")
        
        result_entry["gold_comparison"]["gold_files_compared"] = [str(p) for p in gold_paths]
        result_entry["gold_comparison"]["matches_any_gold"] = matches_any
        result_entry["gold_comparison"]["comparison_details"] = comparison_details
        
        if matches_any:
            print(f"  ✓ Result MATCHES golden standard")
            stats["comparison_correct"] += 1
        else:
            print(f"  ✗ Result does NOT match golden standard")
            stats["comparison_incorrect"] += 1
        
        results.append(result_entry)
    
    # Save results to JSON
    print(f"\nSaving results to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Calculate additional statistics
    if stats['total'] > 0:
        stats['api_success_rate'] = stats['api_success'] / stats['total'] * 100
        stats['exec_success_rate'] = stats['exec_success'] / stats['total'] * 100
        stats['gold_found_rate'] = stats['gold_found'] / stats['total'] * 100
        stats['accuracy'] = stats['comparison_correct'] / stats['total'] * 100
    else:
        stats['api_success_rate'] = 0.0
        stats['exec_success_rate'] = 0.0
        stats['gold_found_rate'] = 0.0
        stats['accuracy'] = 0.0
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    print(f"Total entries processed: {stats['total']}")
    print(f"API calls successful: {stats['api_success']} ({stats['api_success_rate']:.2f}%)")
    print(f"API calls failed: {stats['api_failed']}")
    print(f"SQL executions successful: {stats['exec_success']} ({stats['exec_success_rate']:.2f}%)")
    print(f"SQL executions failed: {stats['exec_failed']}")
    print(f"Golden files found: {stats['gold_found']} ({stats['gold_found_rate']:.2f}%)")
    print(f"Golden files not found: {stats['gold_not_found']}")
    print(f"Results match golden (any variant): {stats['comparison_correct']}")
    print(f"Results do not match golden: {stats['comparison_incorrect']}")
    print(f"\nOverall Accuracy: {stats['accuracy']:.2f}%")
    
    # Save statistics to separate file
    stats_file = output_json_path.replace('.json', '_statistics.json')
    print(f"\nSaving statistics to {stats_file}...")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return {
        "results": results,
        "statistics": stats
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate API-generated SQL against golden results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/spider2/converted/spider2-lite_new_format.json",
        help="Input JSON file with dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/spider2/converted/spider2-lite_api_evaluated.json",
        help="Output JSON file with API results and evaluations"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/infer",
        help="API endpoint URL"
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
        "--gold_result_dir",
        type=str,
        default="../Spider2/spider2-lite/evaluation_suite/gold/exec_result",
        help="Directory containing golden CSV result files"
    )
    parser.add_argument(
        "--eval_metadata_path",
        type=str,
        default="../Spider2/spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl",
        help="Path to evaluation metadata JSONL file"
    )
    parser.add_argument(
        "--result_csv_dir",
        type=str,
        default="./data/spider2/execution_results",
        help="Directory to save execution result CSV files"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="natsql",
        choices=["sql", "natsql"],
        help="Target SQL type"
    )
    parser.add_argument(
        "--api_timeout",
        type=int,
        default=600,
        help="API request timeout in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of API retry attempts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (for testing). Only processes entries with gold_sql_available: true"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(parent_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(parent_dir, args.output) if not os.path.isabs(args.output) else args.output
    database_dir = os.path.join(parent_dir, args.database_dir) if not os.path.isabs(args.database_dir) else args.database_dir
    result_csv_dir = os.path.join(parent_dir, args.result_csv_dir) if not os.path.isabs(args.result_csv_dir) else args.result_csv_dir
    
    # Handle relative paths for gold directories
    if not os.path.isabs(args.gold_result_dir):
        gold_result_dir = os.path.join(parent_dir, args.gold_result_dir)
        if not os.path.exists(gold_result_dir):
            gold_result_dir = args.gold_result_dir
    else:
        gold_result_dir = args.gold_result_dir
    
    if not os.path.isabs(args.eval_metadata_path):
        eval_metadata_path = os.path.join(parent_dir, args.eval_metadata_path)
        if not os.path.exists(eval_metadata_path):
            eval_metadata_path = args.eval_metadata_path
    else:
        eval_metadata_path = args.eval_metadata_path
    
    evaluate_api_results(
        input_json_path=input_path,
        output_json_path=output_path,
        api_url=args.api_url,
        tables_path=args.tables_path,
        database_dir=database_dir,
        gold_result_dir=gold_result_dir,
        eval_metadata_path=eval_metadata_path,
        result_csv_dir=result_csv_dir,
        target_type=args.target_type,
        api_timeout=args.api_timeout,
        max_retries=args.max_retries,
        limit=args.limit
    )
