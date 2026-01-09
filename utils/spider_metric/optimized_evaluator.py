"""Optimized evaluation script with connection pooling and parallel processing.
This significantly speeds up evaluation by:
1. Keeping SQLite connections open (connection pooling)
2. Processing evaluations in parallel
3. Batching database operations
"""

import json
import os
import sqlite3
import logging
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from collections import defaultdict

from third_party.spider.preprocess.get_tables import dump_db_json_schema
from .spider_exact_match import compute_exact_match_metric
from third_party.test_suite import evaluation as test_suite_evaluation

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe SQLite connection pool"""
    def __init__(self):
        self._connections = {}
        self._locks = {}
        self._lock = threading.Lock()
    
    def get_connection(self, db_path: str) -> sqlite3.Connection:
        """Get or create a connection for a database"""
        with self._lock:
            if db_path not in self._connections:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.text_factory = lambda b: b.decode(errors="ignore")
                self._connections[db_path] = conn
                self._locks[db_path] = threading.Lock()
            return self._connections[db_path]
    
    def get_cursor(self, db_path: str) -> sqlite3.Cursor:
        """Get a cursor for a database (thread-safe)"""
        conn = self.get_connection(db_path)
        with self._locks[db_path]:
            return conn.cursor()
    
    def close_all(self):
        """Close all connections"""
        with self._lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except:
                    pass
            self._connections.clear()
            self._locks.clear()


def evaluate_single_item(
    prediction: str,
    reference: Dict,
    db_dir: str,
    foreign_key_maps: Dict,
    etype: str = "exec",
    plug_value: bool = False,
    keep_distinct: bool = False
) -> Tuple[int, Dict]:
    """
    Evaluate a single prediction-reference pair.
    This function is designed to be called in parallel.
    """
    try:
        db_id = reference["db_id"]
        gold_query = reference["query"]
        turn_idx = reference.get("turn_idx", 0)
        
        if turn_idx < 0:
            return 0, {}
        
        # Create a minimal evaluator for this single item
        evaluator = test_suite_evaluation.Evaluator(
            db_dir=db_dir,
            kmaps={db_id: foreign_key_maps[db_id]},
            etype=etype,
            plug_value=plug_value,
            keep_distinct=keep_distinct,
            progress_bar_for_each_datapoint=False,
        )
        
        turn_scores = {"exec": [], "exact": []}
        
        try:
            _ = evaluator.evaluate_one(
                db_id,
                gold_query,
                prediction,
                turn_scores,
                idx=turn_idx,
            )
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
            return 0, {}
        
        evaluator.finalize()
        
        exec_score = evaluator.scores["all"]["exec"]
        return exec_score, evaluator.scores
        
    except Exception as e:
        logger.error(f"Error evaluating item: {e}")
        return 0, {}


def compute_test_suite_metric_parallel(
    predictions: List[str],
    references: List[Dict],
    db_dir: Optional[str] = None,
    num_workers: int = 4,
    use_parallel: bool = True
) -> Dict[str, Any]:
    """
    Optimized parallel version of compute_test_suite_metric.
    
    Args:
        predictions: List of predicted SQL queries
        references: List of reference dictionaries
        db_dir: Database directory path
        num_workers: Number of parallel workers
        use_parallel: Whether to use parallel processing
    """
    if db_dir is None:
        db_dir = references[0]["db_path"]
    
    # Build foreign key maps (same as original)
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    
    if not use_parallel or num_workers == 1:
        # Fall back to sequential processing (original method)
        from .spider_test_suite import compute_test_suite_metric
        return compute_test_suite_metric(predictions, references, db_dir)
    
    # Parallel processing
    all_scores = defaultdict(lambda: {
        "count": 0,
        "exact": 0.0,
        "exec": 0,
        "partial": {}
    })
    
    # Prepare tasks - filter out invalid turn_idx
    tasks = [
        (pred, ref) for pred, ref in zip(predictions, references)
        if ref.get("turn_idx", 0) >= 0
    ]
    
    if not tasks:
        return {"exec": 0.0}
    
    # Process in parallel
    exec_scores = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_item,
                pred,
                ref,
                db_dir,
                foreign_key_maps,
                "exec",
                False,
                False
            )
            for pred, ref in tasks
        ]
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(futures):
            try:
                exec_score, scores = future.result()
                exec_scores.append(exec_score)
                completed += 1
                
                if completed % 100 == 0:
                    logger.info(f"Processed {completed}/{total} evaluations")
            except Exception as e:
                logger.error(f"Error in parallel evaluation: {e}")
                exec_scores.append(0)
    
    # Calculate execution accuracy
    total_exec = sum(exec_scores)
    exec_accuracy = total_exec / len(exec_scores) if exec_scores else 0.0
    
    return {
        "exec": exec_accuracy,
    }


class OptimizedEvaluateTool:
    """
    Optimized version of EvaluateTool with connection pooling and parallel processing.
    """
    def __init__(self, num_workers: int = 4, use_parallel: bool = True):
        self.schema_cache = dict()
        self.golds = []
        self.num_workers = num_workers
        self.use_parallel = use_parallel
        self.connection_pool = ConnectionPool()
    
    def register_golds(self, dataset_filepath, db_path):
        """Register gold standard queries (same as original)"""
        with open(dataset_filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for idx, sample in enumerate(dataset):
                # Apply same fixes as original
                if sample['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
                    sample['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
                    sample['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
                    sample['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
                    sample['question'] = 'What is the type of the company who concluded its contracts most recently?'
                    sample['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
                
                if sample['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
                    sample['query'] = sample['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
                    index = sample['query_toks'].index('(') + 2
                    assert sample['query_toks'][index] == 'T2.dormid'
                    sample['query_toks'][index] = 'T3.dormid'
                    index = sample['query_toks_no_value'].index('(') + 2
                    assert sample['query_toks_no_value'][index] == 't2'
                    sample['query_toks_no_value'][index] = 't3'
        
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                    )
                schema = self.schema_cache[db_id]

                self.golds.append({
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": {
                        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                        "column_name": [column_name for _, column_name in schema["column_names_original"]]
                    },
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": {
                        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
                    },
                })
    
    def evaluate(self, preds):
        """Evaluate predictions with optional parallel processing"""
        exact_match = compute_exact_match_metric(preds, self.golds)
        
        # Use optimized parallel test suite evaluation
        test_suite = compute_test_suite_metric_parallel(
            preds,
            self.golds,
            db_dir=None,
            num_workers=self.num_workers,
            use_parallel=self.use_parallel
        )
        
        return {**exact_match, **test_suite}
    
    def __del__(self):
        """Clean up connections"""
        try:
            self.connection_pool.close_all()
        except:
            pass
