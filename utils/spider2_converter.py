"""
Converter for Spider2 JSONL format to RESDSQL format.
Handles conversion of Spider2 dataset to RESDSQL's expected tables.json and dataset format.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports when running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils.database_adapters import get_database_adapter, DatabaseAdapter


def load_spider2_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load Spider2 JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_db_type_from_spider2(example: Dict[str, Any]) -> str:
    """Extract database type from Spider2 example.
    
    Args:
        example: Spider2 example dictionary
        
    Returns:
        Database type: 'sqlite', 'snowflake', or 'bigquery'
    """
    # Check instance_id prefix - Spider2 uses prefixes to indicate database type
    instance_id = example.get('instance_id', '')
    
    if instance_id.startswith('local'):
        return 'sqlite'
    elif instance_id.startswith('sf') or instance_id.startswith('snow'):
        return 'snowflake'
    elif instance_id.startswith('bq') or instance_id.startswith('bigquery'):
        return 'bigquery'
    
    # Spider2 examples typically have a 'db_type' or 'database_type' field
    # Or we can infer from the database connection info
    db_type = example.get('db_type') or example.get('database_type')
    
    if db_type:
        db_type_lower = db_type.lower()
        if 'sqlite' in db_type_lower:
            return 'sqlite'
        elif 'snowflake' in db_type_lower:
            return 'snowflake'
        elif 'bigquery' in db_type_lower or 'big_query' in db_type_lower:
            return 'bigquery'
    
    # Try to infer from database path or connection info
    db_path = example.get('db_path') or example.get('database_path')
    if db_path:
        if db_path.endswith('.sqlite') or db_path.endswith('.db'):
            return 'sqlite'
        elif 'snowflake' in db_path.lower():
            return 'snowflake'
        elif 'bigquery' in db_path.lower() or 'gcp' in db_path.lower():
            return 'bigquery'
    
    # Default to sqlite if cannot determine
    return 'sqlite'


def get_connection_info_from_spider2(example: Dict[str, Any], db_type: str, db_id: str = None) -> Dict[str, Any]:
    """Extract connection information from Spider2 example.
    
    Args:
        example: Spider2 example dictionary
        db_type: Type of database
        db_id: Database ID (from 'db' field in Spider2)
        
    Returns:
        Dictionary with connection information
    """
    connection_info = {}
    
    if db_type == 'sqlite':
        # SQLite needs db_path
        # Use db_id (from 'db' field) for path construction
        if not db_id:
            db_id = example.get('db') or example.get('db_id') or example.get('database_id')
        
        if db_id:
            # Standard structure: database/{db_id}/{db_id}.sqlite
            # db_id comes from the 'db' field in Spider2
            db_path = f"database/{db_id}/{db_id}.sqlite"
            connection_info['db_path'] = db_path
        else:
            # Fallback to explicit db_path if provided
            db_path = example.get('db_path') or example.get('database_path')
            if db_path:
                connection_info['db_path'] = db_path
        
    elif db_type == 'snowflake':
        # Snowflake connection info
        connection_info['user'] = example.get('snowflake_user') or os.getenv('SNOWFLAKE_USER')
        connection_info['password'] = example.get('snowflake_password') or os.getenv('SNOWFLAKE_PASSWORD')
        connection_info['account'] = example.get('snowflake_account') or os.getenv('SNOWFLAKE_ACCOUNT')
        connection_info['warehouse'] = example.get('snowflake_warehouse') or os.getenv('SNOWFLAKE_WAREHOUSE')
        connection_info['database'] = example.get('snowflake_database') or example.get('db_id') or os.getenv('SNOWFLAKE_DATABASE')
        connection_info['schema'] = example.get('snowflake_schema') or os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
        
    elif db_type == 'bigquery':
        # BigQuery connection info
        connection_info['project_id'] = example.get('bigquery_project') or example.get('project_id') or os.getenv('GOOGLE_CLOUD_PROJECT')
        connection_info['dataset_id'] = example.get('bigquery_dataset') or example.get('dataset_id') or example.get('db_id')
        connection_info['credentials_path'] = example.get('credentials_path') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    return connection_info


def extract_schema_from_database(
    db_id: str,
    db_type: str,
    connection_info: Dict[str, Any],
    adapter: Optional[DatabaseAdapter] = None
) -> Dict[str, Any]:
    """Extract schema from database using adapter.
    
    Args:
        db_id: Database identifier
        db_type: Type of database
        connection_info: Connection information dictionary
        adapter: Optional pre-initialized adapter
        
    Returns:
        Schema dictionary in RESDSQL format
    """
    if adapter is None:
        adapter = get_database_adapter(db_type)
        adapter.connect(connection_info)
    
    try:
        schema = adapter.get_schema(db_id)
        return schema
    finally:
        if adapter:
            adapter.close()


def convert_spider2_to_resdsql_format(
    spider2_data: List[Dict[str, Any]],
    output_tables_path: str,
    output_dataset_path: Optional[str] = None,
    db_base_path: str = "./database"
) -> Dict[str, Any]:
    """Convert Spider2 JSONL data to RESDSQL format.
    
    Args:
        spider2_data: List of Spider2 examples
        output_tables_path: Path to save tables.json file
        output_dataset_path: Optional path to save converted dataset JSON
        db_base_path: Base path for SQLite databases
        
    Returns:
        Dictionary with conversion statistics
    """
    # Group by db_id to extract unique schemas
    db_schemas = {}
    db_types = {}
    connection_infos = {}
    
    # First pass: collect unique databases
    for example in spider2_data:
        # Spider2 uses 'db' field as the database identifier (multiple instance_ids can share the same db)
        db_id = example.get('db') or example.get('db_id') or example.get('database_id')
        if not db_id:
            # Fallback: try to infer from instance_id prefix for cloud databases
            instance_id = example.get('instance_id', '')
            if instance_id.startswith('bq'):
                db_id = example.get('db')  # Should be present for BigQuery
            elif instance_id.startswith('sf') or instance_id.startswith('snow'):
                db_id = example.get('db')  # Should be present for Snowflake
            else:
                db_id = f"db_{len(db_schemas)}"
        
        db_type = extract_db_type_from_spider2(example)
        
        if db_id not in db_schemas:
            db_types[db_id] = db_type
            connection_infos[db_id] = get_connection_info_from_spider2(example, db_type, db_id)
    
    # Second pass: extract schemas for each unique database
    print(f"Extracting schemas for {len(db_types)} unique databases...")
    valid_schemas = {}
    for db_id, db_type in db_types.items():
        print(f"  Extracting schema for {db_id} ({db_type})...")
        try:
            adapter = get_database_adapter(db_type)
            adapter.connect(connection_infos[db_id])
            schema = adapter.get_schema(db_id)  # db_id is the 'db' field value
            adapter.close()
            
            # Validate schema - must have at least one table
            if schema.get("table_names_original") and len(schema["table_names_original"]) > 0:
                valid_schemas[db_id] = schema
                print(f"    ✓ Extracted schema with {len(schema['table_names_original'])} tables")
            else:
                print(f"    ✗ Skipping {db_id}: No tables found (may need cloud credentials)")
        except Exception as e:
            print(f"    ✗ Failed to extract schema for {db_id}: {e}")
            # Don't create empty schema - skip databases we can't access
    
    # Update db_schemas with only valid schemas
    db_schemas = valid_schemas
    
    if not db_schemas:
        raise ValueError("No valid database schemas extracted! Check database connections and credentials.")
    
    print(f"\nSuccessfully extracted {len(db_schemas)} valid database schemas")
    
    # Write tables.json (only valid schemas)
    tables_list = list(db_schemas.values())
    with open(output_tables_path, 'w', encoding='utf-8') as f:
        json.dump(tables_list, f, indent=2, ensure_ascii=False)
    
    # Convert dataset format if output path provided
    # Only include examples that have valid database schemas
    converted_dataset = []
    skipped_count = 0
    if output_dataset_path:
        for example in spider2_data:
            # Spider2 uses 'db' field as the database identifier
            db_id = example.get('db') or example.get('db_id') or example.get('database_id')
            
            # Skip examples without valid schema
            if db_id not in db_schemas:
                skipped_count += 1
                continue
            
            instance_id = example.get('instance_id', '')
            question = example.get('question') or example.get('instruction') or example.get('input')
            
            # Get gold SQL if available
            gold_sql = example.get('gold_sql') or example.get('sql') or example.get('expected_output') or ""
            
            converted_example = {
                "question": question,
                "db_id": db_id,  # Use 'db' field as db_id
                "instance_id": instance_id,  # Keep instance_id for reference
                "query": gold_sql,
                "query_toks": gold_sql.split() if gold_sql else [],
                "query_toks_no_value": gold_sql.lower().split() if gold_sql else []
            }
            converted_dataset.append(converted_example)
        
        if skipped_count > 0:
            print(f"\nWarning: Skipped {skipped_count} examples without valid database schemas")
            print(f"  (These may require cloud database credentials)")
        
        print(f"Writing {len(converted_dataset)} examples to dataset file...")
        with open(output_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(converted_dataset, f, indent=2, ensure_ascii=False)
    
    return {
        "num_databases": len(db_schemas),
        "num_examples": len(converted_dataset) if output_dataset_path else len(spider2_data),
        "num_skipped": skipped_count if output_dataset_path else 0,
        "databases": list(db_schemas.keys())
    }


def convert_spider2_file(
    input_jsonl_path: str,
    output_dir: str,
    dataset_name: str = "spider2"
) -> Dict[str, Any]:
    """Convert a Spider2 JSONL file to RESDSQL format.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_dir: Directory to save output files
        dataset_name: Name for output files
        
    Returns:
        Dictionary with conversion statistics
    """
    # Load Spider2 data
    print(f"Loading Spider2 data from {input_jsonl_path}...")
    spider2_data = load_spider2_jsonl(input_jsonl_path)
    print(f"Loaded {len(spider2_data)} examples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    tables_path = os.path.join(output_dir, f"{dataset_name}_tables.json")
    dataset_path = os.path.join(output_dir, f"{dataset_name}_dataset.json")
    
    # Convert
    stats = convert_spider2_to_resdsql_format(
        spider2_data=spider2_data,
        output_tables_path=tables_path,
        output_dataset_path=dataset_path
    )
    
    print(f"\nConversion complete!")
    print(f"  Tables file: {tables_path}")
    print(f"  Dataset file: {dataset_path}")
    print(f"  Unique databases: {stats['num_databases']}")
    print(f"  Total examples: {stats['num_examples']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Spider2 JSONL to RESDSQL format")
    parser.add_argument("--input", type=str, required=True, help="Input Spider2 JSONL file")
    parser.add_argument("--output_dir", type=str, default="./data/spider2", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="spider2", help="Dataset name for output files")
    
    args = parser.parse_args()
    
    convert_spider2_file(
        input_jsonl_path=args.input,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
