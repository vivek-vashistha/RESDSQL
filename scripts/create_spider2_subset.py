#!/usr/bin/env python3
"""
Create a subset of Spider2 JSONL file for testing.
Filters by number of examples and optionally by database type.
"""

import json
import argparse
from pathlib import Path


def filter_spider2_subset(
    input_file: str,
    output_file: str,
    limit: int = None,
    db_type: str = None,
    instance_ids: list = None
):
    """Create a subset of Spider2 JSONL file.
    
    Args:
        input_file: Path to input Spider2 JSONL file
        output_file: Path to output subset JSONL file
        limit: Maximum number of examples to include (takes first N)
        db_type: Filter by database type ('sqlite', 'bigquery', 'snowflake')
        instance_ids: List of specific instance_ids to include
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    examples = []
    count = 0
    
    print(f"Reading from {input_file}...")
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            example = json.loads(line)
            
            # Filter by instance_id if specified
            if instance_ids:
                if example.get('instance_id') not in instance_ids:
                    continue
            
            # Filter by db_type if specified
            if db_type:
                instance_id = example.get('instance_id', '')
                if db_type == 'sqlite' and not instance_id.startswith('local'):
                    continue
                elif db_type == 'bigquery' and not instance_id.startswith('bq'):
                    continue
                elif db_type == 'snowflake' and not (instance_id.startswith('sf') or instance_id.startswith('snow')):
                    continue
            
            examples.append(example)
            count += 1
            
            # Stop if limit reached
            if limit and count >= limit:
                break
    
    # Write subset
    print(f"Writing {len(examples)} examples to {output_file}...")
    with output_path.open('w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Created subset with {len(examples)} examples")
    
    # Print summary
    if examples:
        db_counts = {}
        for ex in examples:
            db = ex.get('db', 'unknown')
            db_counts[db] = db_counts.get(db, 0) + 1
        
        print(f"\nDatabase distribution:")
        for db, count in sorted(db_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {db}: {count} examples")
        if len(db_counts) > 10:
            print(f"  ... and {len(db_counts) - 10} more databases")


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset of Spider2 JSONL file for testing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Spider2 JSONL file (e.g., data/spider2/spider2-lite.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output subset JSONL file (e.g., data/spider2/spider2-lite-50.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to include (default: all)"
    )
    parser.add_argument(
        "--db_type",
        type=str,
        choices=['sqlite', 'bigquery', 'snowflake'],
        default=None,
        help="Filter by database type (sqlite, bigquery, snowflake)"
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs='+',
        default=None,
        help="Specific instance_ids to include (e.g., local002 local003)"
    )
    
    args = parser.parse_args()
    
    filter_spider2_subset(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        db_type=args.db_type,
        instance_ids=args.instance_ids
    )


if __name__ == "__main__":
    main()
