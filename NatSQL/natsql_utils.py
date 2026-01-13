from .natsql2sql.natsql_parser import create_sql_from_natSQL
from .natsql2sql.natsql2sql import Args

natsql2sql_args = Args()
natsql2sql_args.not_infer_group = True

def natsql_to_sql(natsql, db_id, db_file_path, table_info):
    try:
        query, _, __ = create_sql_from_natSQL(
            natsql, 
            db_id, 
            db_file_path, 
            table_info, 
            sq=None, 
            remove_values=False, 
            remove_groupby_from_natsql=False, 
            args=natsql2sql_args
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Error in natsql_to_sql for db_id={db_id}: {e}")
        import traceback
        traceback.print_exc()
        query = "sql placeholder"
    
    if query == None:
        query = "sql placeholder"
    
    return query