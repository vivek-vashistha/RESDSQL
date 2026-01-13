#!/usr/bin/env python3
"""
Organize Spider2 local SQLite databases into Spider1 format.
Creates database/{db_id}/{db_id}.sqlite structure from flat files.
"""

import json
import os
import shutil
from pathlib import Path

def organize_spider2_sqlite(
    local_sqlite_dir: str = "database/local_sqlite",
    mapping_file: str = "database/local_sqlite/local-map.jsonl",
    target_dir: str = "database"
):
    """Organize Spider2 SQLite databases into proper folder structure.
    
    Uses the 'db' field (database name) as the db_id, not instance_id.
    Multiple instance_ids can share the same database.
    
    Args:
        local_sqlite_dir: Directory containing the flat SQLite files
        mapping_file: Path to the local-map.jsonl file
        target_dir: Target directory (database/) for organized structure
    """
    # Load mapping
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        mapping = json.loads(f.readline().strip())
    
    print(f"Found {len(mapping)} instance_id to db_name mappings")
    
    # Group by db_name (the actual database identifier)
    db_name_to_instances = {}
    for local_id, db_name in mapping.items():
        if db_name not in db_name_to_instances:
            db_name_to_instances[db_name] = []
        db_name_to_instances[db_name].append(local_id)
    
    print(f"Found {len(db_name_to_instances)} unique databases")
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Process each unique database
    for db_name, instance_ids in db_name_to_instances.items():
        source_file = Path(local_sqlite_dir) / f"{db_name}.sqlite"
        
        if not source_file.exists():
            print(f"Warning: Source file not found: {source_file}")
            continue
        
        # Use db_name as db_id (database/{db_name}/{db_name}.sqlite)
        target_db_dir = target_path / db_name
        target_file = target_db_dir / f"{db_name}.sqlite"
        
        # Create directory
        target_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the database file (only once per unique db_name)
        if not target_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"Copied {db_name}.sqlite -> {db_name}/{db_name}.sqlite (used by {len(instance_ids)} instances)")
        else:
            print(f"Skipped {db_name}.sqlite (already exists, used by {len(instance_ids)} instances)")
    
    print(f"\nOrganization complete!")
    print(f"Created {len(db_name_to_instances)} database directories in {target_dir}/")
    print(f"Each database can be used by multiple instance_ids")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize Spider2 SQLite databases")
    parser.add_argument("--local_sqlite_dir", type=str, default="database/local_sqlite",
                        help="Directory containing flat SQLite files")
    parser.add_argument("--mapping_file", type=str, default="database/local_sqlite/local-map.jsonl",
                        help="Path to local-map.jsonl file")
    parser.add_argument("--target_dir", type=str, default="database",
                        help="Target directory for organized structure")
    
    args = parser.parse_args()
    
    organize_spider2_sqlite(
        local_sqlite_dir=args.local_sqlite_dir,
        mapping_file=args.mapping_file,
        target_dir=args.target_dir
    )
