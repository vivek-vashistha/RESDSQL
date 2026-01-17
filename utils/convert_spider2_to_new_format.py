"""
Script to convert spider2-lite_dataset.json to the new format.
Filters for instance_id starting with 'local' and includes gold SQL from evaluation suite.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports when running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """Load the spider2-lite dataset JSON file.
    
    Args:
        input_path: Path to the input JSON file
        
    Returns:
        List of dataset entries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_gold_sql(gold_sql_dir: str, instance_id: str) -> Optional[str]:
    """Read gold SQL from the evaluation suite.
    
    Args:
        gold_sql_dir: Directory containing gold SQL files
        instance_id: Instance ID to look up
        
    Returns:
        SQL string if file exists, None otherwise
    """
    sql_file = os.path.join(gold_sql_dir, f"{instance_id}.sql")
    if os.path.exists(sql_file):
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read SQL file {sql_file}: {e}")
            return None
    return None


def convert_entry(entry: Dict[str, Any], entry_id: int, gold_sql: Optional[str]) -> Dict[str, Any]:
    """Convert a single entry to the new format.
    
    Args:
        entry: Original entry from spider2-lite_dataset.json
        entry_id: Sequential ID for this entry
        gold_sql: Gold SQL string (if available)
        
    Returns:
        Converted entry in new format
    """
    instance_id = entry.get('instance_id', '')
    question = entry.get('question', '')
    db_id = entry.get('db_id', '')
    
    # Determine platform from instance_id prefix
    if instance_id.startswith('local'):
        platform = 'sqlite'
    elif instance_id.startswith('bq') or instance_id.startswith('ga'):
        platform = 'bigquery'
    elif instance_id.startswith('sf'):
        platform = 'snowflake'
    else:
        platform = 'sqlite'  # default
    
    # Build the converted entry
    converted = {
        "id": entry_id,
        "turns": [
            {
                "id": 1,
                "role": "user",
                "input": question,
                "expected_output": None
            }
        ],
        "metadata": {
            "source": "spider2-lite",
            "instance_id": instance_id,
            "platform": platform,
            "database": db_id,
            "gold_sql_available": gold_sql is not None
        },
        "gold_sql": gold_sql if gold_sql else ""
    }
    
    return converted


def convert_dataset(
    input_path: str,
    output_path: str,
    gold_sql_dir: str
) -> Dict[str, Any]:
    """Convert the dataset to the new format.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        gold_sql_dir: Directory containing gold SQL files
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"Loading dataset from {input_path}...")
    dataset = load_dataset(input_path)
    print(f"Loaded {len(dataset)} entries")
    
    # Filter for entries with instance_id starting with 'local'
    local_entries = [
        entry for entry in dataset 
        if entry.get('instance_id', '').startswith('local')
    ]
    print(f"Found {len(local_entries)} entries with instance_id starting with 'local'")
    
    # Convert entries
    converted_entries = []
    missing_sql_count = 0
    found_sql_count = 0
    
    for idx, entry in enumerate(local_entries, start=1):
        instance_id = entry.get('instance_id', '')
        gold_sql = read_gold_sql(gold_sql_dir, instance_id)
        
        if gold_sql:
            found_sql_count += 1
        else:
            missing_sql_count += 1
            print(f"Warning: No SQL file found for instance_id: {instance_id}")
        
        converted_entry = convert_entry(entry, idx, gold_sql)
        converted_entries.append(converted_entry)
    
    # Write output
    print(f"\nWriting {len(converted_entries)} entries to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_entries, f, indent=2, ensure_ascii=False)
    
    stats = {
        "total_entries": len(dataset),
        "local_entries": len(local_entries),
        "converted_entries": len(converted_entries),
        "sql_files_found": found_sql_count,
        "sql_files_missing": missing_sql_count
    }
    
    print(f"\nConversion complete!")
    print(f"  Total entries in input: {stats['total_entries']}")
    print(f"  Local entries filtered: {stats['local_entries']}")
    print(f"  Entries converted: {stats['converted_entries']}")
    print(f"  SQL files found: {stats['sql_files_found']}")
    print(f"  SQL files missing: {stats['sql_files_missing']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert spider2-lite_dataset.json to new format with gold SQL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/spider2/converted/spider2-lite_dataset.json",
        help="Input dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/spider2/converted/spider2-lite_new_format.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--gold_sql_dir",
        type=str,
        default="../Spider2/spider2-lite/evaluation_suite/gold/sql",
        help="Directory containing gold SQL files"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(parent_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(parent_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Handle gold_sql_dir - it might be relative to parent or absolute
    if os.path.isabs(args.gold_sql_dir):
        gold_sql_dir = args.gold_sql_dir
    else:
        # Try relative to parent directory first
        gold_sql_dir = os.path.join(parent_dir, args.gold_sql_dir)
        if not os.path.exists(gold_sql_dir):
            # Try relative to script directory
            gold_sql_dir = os.path.join(script_dir, args.gold_sql_dir)
    
    if not os.path.exists(gold_sql_dir):
        print(f"Warning: Gold SQL directory not found: {gold_sql_dir}")
        print("Please provide the correct path using --gold_sql_dir")
        sys.exit(1)
    
    convert_dataset(input_path, output_path, gold_sql_dir)
