#!/usr/bin/env python3
"""
Quick test script for Spider2 FastAPI server
Tests a few questions from spider2-lite.jsonl
"""

import requests
import json
import sys

API_URL = "http://localhost:8000/infer"
TABLES_PATH = "./data/spider2/converted/spider2-lite_tables.json"

# Test questions from spider2-lite.jsonl
test_cases = [
    {
        "question": "According to the RFM definition document, calculate the average sales per order for each customer within distinct RFM segments, considering only 'delivered' orders.",
        "db_id": "E_commerce"
    },
    {
        "question": "Could you help me calculate the average single career span value in years for all baseball players?",
        "db_id": "Baseball"
    },
    {
        "question": "Can you calculate the median from the highest season goals of each team?",
        "db_id": "f1"
    }
]

def test_api():
    """Test the FastAPI server with Spider2 questions"""
    
    print("=" * 60)
    print("Testing Spider2 FastAPI Server")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Tables path: {TABLES_PATH}")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/docs", timeout=2)
        print("✓ Server is running")
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running!")
        print("  Please start the server first:")
        print("    bash start_api.sh")
        print("    or")
        print("    python api_server_refactored.py")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Testing Questions")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing...")
        print(f"Question: {test_case['question'][:80]}...")
        print(f"Database: {test_case['db_id']}")
        
        try:
            response = requests.post(
                API_URL,
                json={
                    "question": test_case["question"],
                    "db_id": test_case["db_id"],
                    "target_type": "natsql",
                    "spider2_mode": True,
                    "tables_path": TABLES_PATH
                },
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success!")
                print(f"  SQL: {result.get('sql', 'N/A')[:100]}...")
                if result.get('error'):
                    print(f"  Warning: {result['error']}")
            else:
                print(f"✗ Error: {response.status_code}")
                print(f"  {response.text}")
                
        except requests.exceptions.Timeout:
            print("✗ Timeout - request took too long")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print()
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
