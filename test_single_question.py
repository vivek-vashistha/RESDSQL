#!/usr/bin/env python3
"""
Quick test script for a single question against the inference API.

Usage:
    python3 test_single_question.py --question "What year is the movie The Imitation Game from ?" --db-id "imdb" --target-type "natsql"
"""

import json
import argparse
import sys

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Install it with: pip install requests")
    sys.exit(1)


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    if not sql:
        return ""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    return sql


def test_question(api_url: str, question: str, db_id: str, target_type: str = "natsql", expected: str = None):
    """Test a single question against the API."""
    print(f"Testing question: {question}")
    print(f"Database: {db_id}")
    print(f"Target type: {target_type}")
    print("-" * 80)
    
    payload = {
        "question": question,
        "db_id": db_id,
        "target_type": target_type
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"Generated SQL: {result.get('sql', 'N/A')}")
        print(f"Execution success: {result.get('execution_success', 'N/A')}")
        
        if expected:
            print(f"Expected SQL: {expected}")
            generated_norm = normalize_sql(result.get('sql', ''))
            expected_norm = normalize_sql(expected)
            match = generated_norm == expected_norm
            print(f"Match: {'✓ YES' if match else '✗ NO'}")
            if not match:
                print(f"Normalized Generated: {generated_norm}")
                print(f"Normalized Expected: {expected_norm}")
        
        if result.get('input_sequence'):
            print(f"\nInput sequence: {result['input_sequence'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test a single question against the inference API")
    parser.add_argument("--api-url", default="http://localhost:8000/infer", help="API endpoint URL")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--db-id", required=True, help="Database identifier")
    parser.add_argument("--target-type", choices=["sql", "natsql"], default="natsql", help="Target query type")
    parser.add_argument("--expected", help="Expected SQL/NatSQL query for comparison")
    
    args = parser.parse_args()
    
    test_question(
        api_url=args.api_url,
        question=args.question,
        db_id=args.db_id,
        target_type=args.target_type,
        expected=args.expected
    )


if __name__ == "__main__":
    main()
