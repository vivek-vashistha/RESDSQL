#!/usr/bin/env python3
"""
Dataset Format Generation Script

Converts Spider dataset format to the target format used for evaluation and testing.
Generates both SQL and NatSQL versions in separate JSON files.
"""

import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def load_natsql_data(natsql_path):
    """Load NatSQL data if available"""
    if not os.path.exists(natsql_path):
        return None
    try:
        with open(natsql_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load NatSQL data from {natsql_path}: {e}")
        return None


def find_natsql_file(input_path):
    """Try to find corresponding NatSQL file"""
    input_path_obj = Path(input_path)
    JSON_EXT = ".json"
    NATSQL_SUFFIX = "-natsql.json"
    
    # Try common NatSQL file locations
    possible_paths = [
        # Same directory with -natsql suffix
        input_path_obj.parent / f"{input_path_obj.stem}-natsql{input_path_obj.suffix}",
        # NatSQL directory
        Path("NatSQL/NatSQLv1_6") / input_path_obj.name.replace(JSON_EXT, NATSQL_SUFFIX),
        # Preprocessed data directory
        Path("data/preprocessed_data") / input_path_obj.name.replace(JSON_EXT, NATSQL_SUFFIX),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def generate_output_entry(spider_entry, natsql_entry, instance_id, target_type):
    """Generate a single output entry"""
    question = spider_entry.get("question", "").strip()
    db_id = spider_entry.get("db_id", "")
    
    if not question or not db_id:
        return None
    
    # Get the expected output based on target_type
    if target_type == "sql":
        expected_output = spider_entry.get("query", "").strip()
    else:  # natsql
        if natsql_entry and "NatSQL" in natsql_entry:
            expected_output = natsql_entry["NatSQL"].strip()
        else:
            # If no NatSQL available, skip this entry
            return None
    
    if not expected_output:
        return None
    
    return {
        "turns": [
            {
                "input": question,
                "expected_output": expected_output,
                "metadata": {
                    "instance_id": str(instance_id),
                    "database": db_id,
                    "target_type": target_type
                }
            }
        ]
    }


def generate_dataset(input_path, output_path, natsql_path=None, target_type="sql", start_index=0, limit=None):
    """Generate dataset in the target format"""
    
    # Load input data
    print(f"Loading input data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        spider_data = json.load(f)
    
    # Load NatSQL data if available
    natsql_data = None
    if natsql_path:
        print(f"Loading NatSQL data from {natsql_path}...")
        natsql_data = load_natsql_data(natsql_path)
    elif target_type == "natsql":
        # Try to find NatSQL file automatically
        auto_natsql_path = find_natsql_file(input_path)
        if auto_natsql_path:
            print(f"Found NatSQL file at {auto_natsql_path}")
            natsql_data = load_natsql_data(auto_natsql_path)
        else:
            print("Warning: NatSQL file not found. NatSQL entries will be skipped.")
    
    # Process entries
    output_data = []
    skipped = 0
    
    # Determine range to process
    end_index = len(spider_data) if limit is None else min(start_index + limit, len(spider_data))
    entries_to_process = spider_data[start_index:end_index]
    
    print(f"Processing {len(entries_to_process)} entries (starting from index {start_index})...")
    
    for idx, spider_entry in enumerate(tqdm(entries_to_process, desc="Processing")):
        instance_id = start_index + idx
        
        # Get corresponding NatSQL entry if available
        natsql_entry = None
        if natsql_data and idx < len(natsql_data):
            natsql_entry = natsql_data[idx]
        
        # Generate output entry
        output_entry = generate_output_entry(spider_entry, natsql_entry, instance_id, target_type)
        
        if output_entry:
            output_data.append(output_entry)
        else:
            skipped += 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write output
    print(f"Writing output to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\nConversion complete!")
    print(f"  Total entries processed: {len(entries_to_process)}")
    print(f"  Entries generated: {len(output_data)}")
    print(f"  Entries skipped: {skipped}")
    
    if output_data:
        databases = {entry["turns"][0]["metadata"]["database"] for entry in output_data}
        print(f"  Unique databases: {len(databases)}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert Spider dataset format to target format (SQL/NatSQL)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Spider dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--natsql_input",
        type=str,
        default=None,
        help="Path to NatSQL dataset JSON file (optional, will try to auto-detect if not provided)"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="sql",
        choices=["sql", "natsql"],
        help="Target type: 'sql' or 'natsql' (default: 'sql')"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for instance_id (default: 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to process (default: all)"
    )
    parser.add_argument(
        "--generate_both",
        action="store_true",
        help="Generate both SQL and NatSQL versions automatically"
    )
    
    args = parser.parse_args()
    
    # If generate_both is set, generate both versions
    if args.generate_both:
        JSON_EXT = ".json"
        SQL_SUFFIX = "_sql.json"
        NATSQL_SUFFIX = "_natsql.json"
        
        # Generate SQL version
        if args.output.endswith(SQL_SUFFIX):
            sql_output = args.output
        else:
            sql_output = args.output.replace(JSON_EXT, SQL_SUFFIX)
        
        print("=" * 60)
        print("Generating SQL version...")
        print("=" * 60)
        generate_dataset(
            args.input,
            sql_output,
            natsql_path=args.natsql_input,
            target_type="sql",
            start_index=args.start_index,
            limit=args.limit
        )
        
        # Generate NatSQL version
        if args.output.endswith(SQL_SUFFIX):
            natsql_output = args.output.replace(SQL_SUFFIX, NATSQL_SUFFIX)
        else:
            natsql_output = args.output.replace(JSON_EXT, NATSQL_SUFFIX)
        print("\n" + "=" * 60)
        print("Generating NatSQL version...")
        print("=" * 60)
        generate_dataset(
            args.input,
            natsql_output,
            natsql_path=args.natsql_input,
            target_type="natsql",
            start_index=args.start_index,
            limit=args.limit
        )
        
        print("\n" + "=" * 60)
        print("Both versions generated successfully!")
        print(f"  SQL output: {sql_output}")
        print(f"  NatSQL output: {natsql_output}")
        print("=" * 60)
    else:
        # Generate single version
        generate_dataset(
            args.input,
            args.output,
            natsql_path=args.natsql_input,
            target_type=args.target_type,
            start_index=args.start_index,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
