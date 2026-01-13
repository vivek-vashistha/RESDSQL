"""
Database adapters for different database types (SQLite, Snowflake, BigQuery).
Provides a unified interface for schema extraction, sample data retrieval, and query execution.
"""

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import json

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""
    
    @abstractmethod
    def connect(self, connection_info: Dict[str, Any]) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Extract table/column schema for a database.
        
        Returns:
            Dictionary with schema information in RESDSQL format:
            {
                "db_id": str,
                "table_names_original": List[str],
                "table_names": List[str],
                "column_names_original": List[Tuple[int, str]],
                "column_names": List[Tuple[int, str]],
                "column_types": List[str],
                "primary_keys": List[int],
                "foreign_keys": List[Tuple[int, int]]
            }
        """
        pass
    
    @abstractmethod
    def get_sample_data(self, table_name: str, column_name: str, limit: int = 5) -> List[str]:
        """Get sample values from a column for schema linking.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            limit: Maximum number of samples to return
            
        Returns:
            List of sample values as strings
        """
        pass
    
    @abstractmethod
    def execute_query(self, sql: str) -> List[Tuple]:
        """Execute a SQL query and return results.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of tuples representing query results
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """Adapter for SQLite databases."""
    
    def __init__(self):
        self.connection = None
        self.db_path = None
    
    def connect(self, connection_info: Dict[str, Any]) -> None:
        """Connect to SQLite database.
        
        Args:
            connection_info: Dictionary with 'db_path' key pointing to database file
        """
        self.db_path = connection_info.get('db_path')
        if not self.db_path or not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.text_factory = lambda b: b.decode(errors="ignore")
    
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Extract schema from SQLite database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = self.connection.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
        
        table_names_original = []
        table_names = []
        column_names_original = [(-1, "*")]
        column_names = [(-1, "*")]
        column_types = ["text"]  # Placeholder for *
        primary_keys = []
        foreign_keys = []
        
        table_id_map = {}
        column_idx = 1  # Start at 1 because 0 is reserved for *
        
        for table_idx, table_name in enumerate(tables):
            table_names_original.append(table_name)
            table_names.append(table_name.lower())
            table_id_map[table_name] = table_idx
            
            # Get columns for this table
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            pk_columns = []
            for col_info in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col_info
                
                column_names_original.append((table_idx, col_name))
                column_names.append((table_idx, col_name.lower()))
                
                # Map SQLite types to standard types
                if 'int' in col_type.lower():
                    col_type_str = "number"
                elif 'real' in col_type.lower() or 'float' in col_type.lower() or 'double' in col_type.lower():
                    col_type_str = "number"
                elif 'text' in col_type.lower() or 'char' in col_type.lower() or 'varchar' in col_type.lower():
                    col_type_str = "text"
                elif 'blob' in col_type.lower():
                    col_type_str = "text"
                else:
                    col_type_str = "text"
                
                column_types.append(col_type_str)
                
                if is_pk:
                    pk_columns.append(column_idx)
                    primary_keys.append(column_idx)
                
                column_idx += 1
        
        # Get foreign keys
        for table_name in tables:
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            fks = cursor.fetchall()
            
            for fk_info in fks:
                # fk_info: (id, seq, table, from, to, on_update, on_delete, match)
                source_table = table_name
                source_col = fk_info[3]
                target_table = fk_info[2]
                target_col = fk_info[4]
                
                # Find column indices
                source_col_idx = None
                target_col_idx = None
                
                for idx, (tbl_idx, col_name) in enumerate(column_names_original):
                    if tbl_idx == table_id_map.get(source_table) and col_name == source_col:
                        source_col_idx = idx
                    if tbl_idx == table_id_map.get(target_table) and col_name == target_col:
                        target_col_idx = idx
                
                if source_col_idx and target_col_idx:
                    foreign_keys.append((source_col_idx, target_col_idx))
        
        return {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "table_names": table_names,
            "column_names_original": column_names_original,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    
    def get_sample_data(self, table_name: str, column_name: str, limit: int = 5) -> List[str]:
        """Get sample data from SQLite column."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = self.connection.cursor()
        try:
            query = f'SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT {limit}'
            cursor.execute(query)
            results = cursor.fetchall()
            return [str(row[0]) for row in results if row[0] is not None]
        except Exception as e:
            print(f"Error getting sample data from {table_name}.{column_name}: {e}")
            return []
    
    def execute_query(self, sql: str) -> List[Tuple]:
        """Execute SQL query on SQLite database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def close(self) -> None:
        """Close SQLite connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


class SnowflakeAdapter(DatabaseAdapter):
    """Adapter for Snowflake databases."""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self, connection_info: Dict[str, Any]) -> None:
        """Connect to Snowflake database.
        
        Args:
            connection_info: Dictionary with connection parameters:
                - user: Snowflake username
                - password: Snowflake password
                - account: Snowflake account identifier
                - warehouse: Warehouse name
                - database: Database name
                - schema: Schema name
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python not installed. Install with: pip install snowflake-connector-python")
        
        # Get credentials from connection_info or environment variables
        user = connection_info.get('user') or os.getenv('SNOWFLAKE_USER')
        password = connection_info.get('password') or os.getenv('SNOWFLAKE_PASSWORD')
        account = connection_info.get('account') or os.getenv('SNOWFLAKE_ACCOUNT')
        warehouse = connection_info.get('warehouse') or os.getenv('SNOWFLAKE_WAREHOUSE')
        database = connection_info.get('database') or os.getenv('SNOWFLAKE_DATABASE')
        schema = connection_info.get('schema') or os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
        
        if not all([user, password, account, warehouse, database]):
            raise ValueError("Missing required Snowflake connection parameters")
        
        self.connection = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        self.cursor = self.connection.cursor()
    
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Extract schema from Snowflake database."""
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        # Get all tables in current schema
        self.cursor.execute("SHOW TABLES")
        tables = [row[1] for row in self.cursor.fetchall()]
        
        table_names_original = []
        table_names = []
        column_names_original = [(-1, "*")]
        column_names = [(-1, "*")]
        column_types = ["text"]
        primary_keys = []
        foreign_keys = []
        
        table_id_map = {}
        column_idx = 1
        
        for table_idx, table_name in enumerate(tables):
            table_names_original.append(table_name)
            table_names.append(table_name.lower())
            table_id_map[table_name] = table_idx
            
            # Get columns for this table
            self.cursor.execute(f"DESCRIBE TABLE {table_name}")
            columns = self.cursor.fetchall()
            
            for col_info in columns:
                col_name = col_info[0]
                col_type = col_info[1]
                
                column_names_original.append((table_idx, col_name))
                column_names.append((table_idx, col_name.lower()))
                
                # Map Snowflake types to standard types
                if 'NUMBER' in col_type.upper() or 'INT' in col_type.upper() or 'FLOAT' in col_type.upper():
                    col_type_str = "number"
                else:
                    col_type_str = "text"
                
                column_types.append(col_type_str)
                column_idx += 1
        
        # Note: Snowflake doesn't enforce primary/foreign keys in the same way
        # This would need to be extracted from constraints if available
        
        return {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "table_names": table_names,
            "column_names_original": column_names_original,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    
    def get_sample_data(self, table_name: str, column_name: str, limit: int = 5) -> List[str]:
        """Get sample data from Snowflake column."""
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        try:
            query = f'SELECT DISTINCT "{column_name}" FROM "{table_name}" LIMIT {limit}'
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [str(row[0]) for row in results if row[0] is not None]
        except Exception as e:
            print(f"Error getting sample data from {table_name}.{column_name}: {e}")
            return []
    
    def execute_query(self, sql: str) -> List[Tuple]:
        """Execute SQL query on Snowflake database."""
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def close(self) -> None:
        """Close Snowflake connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.connection = None
        self.cursor = None


class BigQueryAdapter(DatabaseAdapter):
    """Adapter for Google BigQuery databases."""
    
    def __init__(self):
        self.client = None
        self.project_id = None
        self.dataset_id = None
    
    def connect(self, connection_info: Dict[str, Any]) -> None:
        """Connect to BigQuery database.
        
        Args:
            connection_info: Dictionary with connection parameters:
                - project_id: GCP project ID
                - dataset_id: BigQuery dataset ID
                - credentials_path: Path to service account JSON (optional, can use env var)
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")
        
        credentials_path = connection_info.get('credentials_path') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=credentials)
        else:
            # Use default credentials
            self.client = bigquery.Client()
        
        self.project_id = connection_info.get('project_id') or self.client.project
        self.dataset_id = connection_info.get('dataset_id')
    
    def get_schema(self, db_id: str) -> Dict[str, Any]:
        """Extract schema from BigQuery dataset."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        if not self.dataset_id:
            raise ValueError("dataset_id not specified")
        
        dataset_ref = self.client.dataset(self.dataset_id, project=self.project_id)
        tables = list(self.client.list_tables(dataset_ref))
        
        table_names_original = []
        table_names = []
        column_names_original = [(-1, "*")]
        column_names = [(-1, "*")]
        column_types = ["text"]
        primary_keys = []
        foreign_keys = []
        
        table_id_map = {}
        column_idx = 1
        
        for table_idx, table in enumerate(tables):
            table_name = table.table_id
            table_names_original.append(table_name)
            table_names.append(table_name.lower())
            table_id_map[table_name] = table_idx
            
            # Get table schema
            table_ref = dataset_ref.table(table_name)
            table_obj = self.client.get_table(table_ref)
            
            for field in table_obj.schema:
                col_name = field.name
                col_type = field.field_type
                
                column_names_original.append((table_idx, col_name))
                column_names.append((table_idx, col_name.lower()))
                
                # Map BigQuery types to standard types
                if col_type in ['INTEGER', 'INT64', 'FLOAT', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC']:
                    col_type_str = "number"
                else:
                    col_type_str = "text"
                
                column_types.append(col_type_str)
                column_idx += 1
        
        # BigQuery doesn't enforce primary/foreign keys in the same way
        # This would need to be extracted from metadata if available
        
        return {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "table_names": table_names,
            "column_names_original": column_names_original,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    
    def get_sample_data(self, table_name: str, column_name: str, limit: int = 5) -> List[str]:
        """Get sample data from BigQuery column."""
        if not self.client or not self.dataset_id:
            raise RuntimeError("Database not connected")
        
        try:
            query = f'''
            SELECT DISTINCT `{column_name}`
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            WHERE `{column_name}` IS NOT NULL
            LIMIT {limit}
            '''
            results = self.client.query(query).result()
            return [str(row[0]) for row in results]
        except Exception as e:
            print(f"Error getting sample data from {table_name}.{column_name}: {e}")
            return []
    
    def execute_query(self, sql: str) -> List[Tuple]:
        """Execute SQL query on BigQuery database."""
        if not self.client:
            raise RuntimeError("Database not connected")
        
        query_job = self.client.query(sql)
        results = query_job.result()
        return [tuple(row.values()) for row in results]
    
    def close(self) -> None:
        """Close BigQuery client (no persistent connection to close)."""
        self.client = None


def get_database_adapter(db_type: str) -> DatabaseAdapter:
    """Factory function to get appropriate database adapter.
    
    Args:
        db_type: Type of database ('sqlite', 'snowflake', 'bigquery')
        
    Returns:
        Appropriate DatabaseAdapter instance
    """
    db_type_lower = db_type.lower()
    
    if db_type_lower == 'sqlite':
        return SQLiteAdapter()
    elif db_type_lower == 'snowflake':
        return SnowflakeAdapter()
    elif db_type_lower == 'bigquery':
        return BigQueryAdapter()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
