"""
FastAPI server for RESDSQL Text-to-SQL inference
"""
import os
import sys
import json
import torch
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add NatSQL directory to path for natsql2sql imports
natsql_dir = os.path.join(current_dir, "NatSQL")
if os.path.exists(natsql_dir) and natsql_dir not in sys.path:
    sys.path.insert(0, natsql_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from tokenizers import AddedToken
from transformers.trainer_utils import set_seed

# Import RESDSQL modules
from preprocessing import get_db_schemas, get_db_contents
from schema_item_classifier import prepare_batch_inputs_and_labels
from utils.classifier_model import MyClassifier
from utils.load_dataset import ColumnAndTableClassifierDataset, Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls
from text2sql_data_generator import prepare_input_and_output
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast
import preprocessing
import text2sql_data_generator

# For NatSQL
try:
    from NatSQL.natsql_utils import natsql_to_sql
    # transform_tables is not always needed, make it optional
    try:
        from NatSQL.table_transform import transform_tables
    except ImportError:
        transform_tables = None  # Not critical for inference
    NATSQL_AVAILABLE = True
except ImportError as e:
    NATSQL_AVAILABLE = False
    print(f"Warning: NatSQL not available. Only SQL mode will work. Error: {e}")

app = FastAPI(title="RESDSQL Text-to-SQL API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
schema_classifier_model = None
schema_classifier_tokenizer = None
text2sql_model = None
text2sql_tokenizer = None
tables_dict = None
all_db_infos = None
device = None

# Request/Response models
class InferenceRequest(BaseModel):
    question: str
    db_id: str
    target_type: Optional[str] = "natsql"  # "sql" or "natsql"
    use_contents: Optional[bool] = True
    add_fk_info: Optional[bool] = True
    topk_table_num: Optional[int] = 4
    topk_column_num: Optional[int] = 5
    num_beams: Optional[int] = 8
    num_return_sequences: Optional[int] = 8

class InferenceResponse(BaseModel):
    sql: str
    input_sequence: Optional[str] = None
    execution_success: Optional[bool] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str

def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_models(
    model_scale: str = "base",
    target_type: str = "natsql",
    schema_classifier_path: Optional[str] = None,
    text2sql_model_path: Optional[str] = None,
    tables_path: Optional[str] = None,
    db_path: Optional[str] = None
):
    """Load models and prepare for inference"""
    global schema_classifier_model, schema_classifier_tokenizer
    global text2sql_model, text2sql_tokenizer, tables_dict, all_db_infos, device
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Set device environment
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Determine model paths
    if schema_classifier_path is None:
        if target_type == "natsql":
            schema_classifier_path = "./models/text2natsql_schema_item_classifier"
        else:
            schema_classifier_path = "./models/text2sql_schema_item_classifier"
    
    if text2sql_model_path is None:
        if target_type == "natsql":
            if model_scale == "base":
                text2sql_model_path = "./models/text2natsql-t5-base/checkpoint-14352"
            elif model_scale == "large":
                text2sql_model_path = "./models/text2natsql-t5-large/checkpoint-21216"
            elif model_scale == "3b":
                text2sql_model_path = "./models/text2natsql-t5-3b/checkpoint-78302"
            else:
                raise ValueError(f"Unknown model scale: {model_scale}")
        else:
            if model_scale == "base":
                text2sql_model_path = "./models/text2sql-t5-base/checkpoint-39312"
            elif model_scale == "large":
                text2sql_model_path = "./models/text2sql-t5-large/checkpoint-30576"
            elif model_scale == "3b":
                text2sql_model_path = "./models/text2sql-t5-3b/checkpoint-78302"
            else:
                raise ValueError(f"Unknown model scale: {model_scale}")
    
    if tables_path is None:
        tables_path = "./data/spider/tables.json"
    
    if db_path is None:
        db_path = "./database"
    
    # Load schema classifier
    print(f"Loading schema classifier from {schema_classifier_path}...")
    if "xlm" in schema_classifier_path.lower() or "mt5" in schema_classifier_path.lower():
        schema_classifier_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            schema_classifier_path if os.path.exists(f"{schema_classifier_path}/tokenizer.json") 
            else "xlm-roberta-large"
        )
    else:
        schema_classifier_tokenizer = RobertaTokenizerFast.from_pretrained(
            schema_classifier_path if os.path.exists(f"{schema_classifier_path}/tokenizer.json")
            else "roberta-large"
        )
    
    schema_classifier_tokenizer.add_tokens([AddedToken("[FK]")])
    
    schema_classifier_model = MyClassifier(
        model_name_or_path=schema_classifier_path if os.path.isdir(schema_classifier_path) 
        else ("xlm-roberta-large" if "xlm" in schema_classifier_path.lower() else "roberta-large"),
        vocab_size=len(schema_classifier_tokenizer),
        mode="test"
    )
    
    if device == "cuda":
        schema_classifier_model = schema_classifier_model.cuda()
    elif device == "mps":
        schema_classifier_model = schema_classifier_model.to("mps")
    
    schema_classifier_model.eval()
    
    # Load text2sql model
    print(f"Loading text2sql model from {text2sql_model_path}...")
    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        text2sql_model_path,
        add_prefix_space=True
    )
    
    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    model_class = MT5ForConditionalGeneration if "mt5" in text2sql_model_path.lower() else T5ForConditionalGeneration
    text2sql_model = model_class.from_pretrained(text2sql_model_path)
    
    if device == "cuda":
        text2sql_model = text2sql_model.cuda()
    elif device == "mps":
        text2sql_model = text2sql_model.to("mps")
    
    text2sql_model.eval()
    
    # Load tables info
    print(f"Loading tables from {tables_path}...")
    with open(tables_path, 'r', encoding='utf-8') as f:
        all_db_infos = json.load(f)
    
    # Load tables for NatSQL if needed
    if target_type == "natsql" and NATSQL_AVAILABLE:
        tables_for_natsql_path = "./data/preprocessed_data/test_tables_for_natsql.json"
        if os.path.exists(tables_for_natsql_path):
            with open(tables_for_natsql_path, 'r', encoding='utf-8') as f:
                tables_list = json.load(f)
            tables_dict = {t["db_id"]: t for t in tables_list}
        else:
            tables_dict = {}
            print("Warning: NatSQL tables file not found. NatSQL inference may not work correctly.")
    else:
        tables_dict = {}
    
    print("Models loaded successfully!")
    return device

def preprocess_single_question(
    question: str,
    db_id: str,
    all_db_infos: List[Dict],
    db_path: str,
    target_type: str = "natsql"
):
    """Preprocess a single question for inference"""
    db_schemas = get_db_schemas(all_db_infos)
    
    # Find the database info
    db_info = None
    for db in all_db_infos:
        if db["db_id"] == db_id:
            db_info = db
            break
    
    if db_info is None:
        raise ValueError(f"Database {db_id} not found in tables.json")
    
    question = question.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace("\u201d", "'").strip()
    
    # Build db_schema structure
    preprocessed_data = {
        "question": question,
        "db_id": db_id,
        "sql": "",
        "norm_sql": "",
        "sql_skeleton": "",
        "natsql": "",
        "norm_natsql": "",
        "natsql_skeleton": "",
        "db_schema": [],
        "pk": db_schemas[db_id]["pk"],
        "fk": db_schemas[db_id]["fk"],
        "table_labels": [],
        "column_labels": []
    }
    
    table_names_original = db_info["table_names_original"]
    table_names = db_info["table_names"]
    column_names_original = db_info["column_names_original"]
    column_names = db_info["column_names"]
    column_types = db_info["column_types"]
    
    for table_id, table_name_original in enumerate(table_names_original):
        if table_id == 0:  # Skip the special "*" table
            continue
        
        table_name = table_names[table_id]
        table_schema = {
            "table_name_original": table_name_original.lower(),
            "table_name": table_name.lower(),
            "column_names_original": [],
            "column_names": [],
            "column_types": [],
            "db_contents": []
        }
        
        # Collect columns for this table
        table_columns = []
        for col_id, (table_idx, col_name_original) in enumerate(column_names_original):
            if table_idx == table_id:
                col_name = column_names[col_id][1] if column_names[col_id][1] else ""
                col_type = column_types[col_id]
                table_columns.append((col_name_original, col_name, col_type))
        
        # Get database contents for all columns in this table at once
        column_names_list = [col[0] for col in table_columns]
        if column_names_list:
            db_contents_list = get_db_contents(
                question,
                table_name_original,
                column_names_list,
                db_id,
                db_path
            )
        else:
            db_contents_list = []
        
        # Build table schema
        for idx, (col_name_original, col_name, col_type) in enumerate(table_columns):
            table_schema["column_names_original"].append(col_name_original.lower())
            table_schema["column_names"].append(col_name.lower())
            table_schema["column_types"].append(col_type)
            table_schema["db_contents"].append(db_contents_list[idx] if idx < len(db_contents_list) else [])
        
        if table_schema["column_names_original"]:  # Only add if table has columns
            preprocessed_data["db_schema"].append(table_schema)
            preprocessed_data["table_labels"].append(0)
            preprocessed_data["column_labels"].append([0] * len(table_schema["column_names_original"]))
    
    return preprocessed_data

def classify_schema_items(
    preprocessed_data: Dict,
    use_contents: bool = True,
    add_fk_info: bool = True
):
    """Classify schema items using the schema classifier"""
    global schema_classifier_model, schema_classifier_tokenizer, device
    
    # Create a dataset-like structure
    question = preprocessed_data["question"]
    table_names = [t["table_name"] for t in preprocessed_data["db_schema"]]
    column_infos = []
    
    for table in preprocessed_data["db_schema"]:
        column_info_list = []
        for col_name, contents in zip(
            table["column_names"],
            table["db_contents"]
        ):
            if use_contents and contents:
                # Limit content to first 5 items to avoid very long sequences
                content_str = " , ".join(contents[:5])
                column_info = f"{table['table_name_original']}.{col_name} ( {content_str} )"
            else:
                column_info = f"{table['table_name_original']}.{col_name}"
            column_info_list.append(column_info)
        column_infos.append(column_info_list)
    
    # Prepare batch - need to match the expected format
    # Format: (question, table_names, table_labels, column_infos, column_labels)
    table_labels = [0] * len(table_names)
    column_labels = []
    for col_info_list in column_infos:
        column_labels.extend([0] * len(col_info_list))
    
    batch = [(question, table_names, table_labels, column_infos, column_labels)]
    
    encoder_input_ids, encoder_input_attention_mask, \
        batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
        batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
        batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, schema_classifier_tokenizer)
    
    # Move to device
    if device == "cuda":
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
    elif device == "mps":
        encoder_input_ids = encoder_input_ids.to("mps")
        encoder_input_attention_mask = encoder_input_attention_mask.to("mps")
    
    # Get predictions
    with torch.no_grad():
        model_outputs = schema_classifier_model(
            encoder_input_ids,
            encoder_input_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table
        )
    
    # Extract probabilities
    table_probs = torch.nn.functional.softmax(model_outputs["batch_table_name_cls_logits"][0], dim=1)[:, 1].cpu().tolist()
    column_probs = torch.nn.functional.softmax(model_outputs["batch_column_info_cls_logits"][0], dim=1)[:, 1].cpu().tolist()
    
    # Update preprocessed_data with probabilities
    preprocessed_data["table_probs"] = table_probs
    preprocessed_data["column_probs"] = column_probs
    
    return preprocessed_data

def generate_input_sequence(
    preprocessed_data: Dict,
    target_type: str = "natsql",
    use_contents: bool = True,
    add_fk_info: bool = True,
    topk_table_num: int = 4,
    topk_column_num: int = 5
):
    """Generate input sequence for text2sql model"""
    # Rank tables and columns by probability
    table_probs = preprocessed_data.get("table_probs", [0] * len(preprocessed_data["db_schema"]))
    column_probs = preprocessed_data.get("column_probs", [])
    
    # Sort tables by probability
    table_indices = sorted(range(len(table_probs)), key=lambda i: table_probs[i], reverse=True)
    topk_tables = table_indices[:topk_table_num]
    
    # Build ranked schema
    ranked_schema = []
    col_idx = 0
    for table_idx in topk_tables:
        if table_idx >= len(preprocessed_data["db_schema"]):
            continue
        table = preprocessed_data["db_schema"][table_idx]
        num_cols = len(table["column_names_original"])
        table_col_probs = column_probs[col_idx:col_idx+num_cols] if col_idx < len(column_probs) else [0] * num_cols
        
        # Sort columns by probability
        col_indices = sorted(range(len(table_col_probs)), key=lambda i: table_col_probs[i], reverse=True)
        topk_cols = col_indices[:min(topk_column_num, len(col_indices))]
        
        ranked_table = {
            "table_name_original": table["table_name_original"],
            "column_names_original": [table["column_names_original"][i] for i in topk_cols],
            "db_contents": [table["db_contents"][i] for i in topk_cols]
        }
        ranked_schema.append(ranked_table)
        col_idx += num_cols
    
    # Create ranked data structure
    ranked_data = {
        "question": preprocessed_data["question"],
        "db_schema": ranked_schema,
        "fk": preprocessed_data["fk"],
        "norm_sql": "",
        "norm_natsql": "",
        "sql_skeleton": "",
        "natsql_skeleton": ""
    }
    
    # Use the prepare_input_and_output function
    class Opt:
        def __init__(self, use_contents_val, add_fk_info_val, target_type_val):
            self.use_contents = use_contents_val
            self.add_fk_info = add_fk_info_val
            self.output_skeleton = True
            self.target_type = target_type_val
    
    opt = Opt(use_contents, add_fk_info, target_type)
    input_sequence, _ = prepare_input_and_output(opt, ranked_data)
    
    return input_sequence, ranked_data

def generate_sql(
    input_sequence: str,
    db_id: str,
    db_path: str,
    target_type: str = "natsql",
    num_beams: int = 8,
    num_return_sequences: int = 8,
    ranked_data: Optional[Dict] = None
):
    """Generate SQL from input sequence"""
    global text2sql_model, text2sql_tokenizer, tables_dict, device
    
    # Ensure tables_dict is loaded for NatSQL mode
    if target_type == "natsql":
        if tables_dict is None or len(tables_dict) == 0:
            # Try to load tables_dict if not loaded
            if NATSQL_AVAILABLE:
                tables_for_natsql_path = "./data/preprocessed_data/test_tables_for_natsql.json"
                if os.path.exists(tables_for_natsql_path):
                    with open(tables_for_natsql_path, 'r', encoding='utf-8') as f:
                        tables_list = json.load(f)
                    tables_dict = {t["db_id"]: t for t in tables_list}
                else:
                    raise ValueError("NatSQL tables file not found. Cannot use NatSQL mode.")
            else:
                raise ValueError("NatSQL module not available. Cannot use NatSQL mode.")
        
        # Check if db_id exists in tables_dict
        if db_id not in tables_dict:
            available = list(tables_dict.keys())[:10]
            raise KeyError(f"Database '{db_id}' not found in NatSQL tables dictionary. Available databases (sample): {available}...")
    
    # Tokenize input
    tokenized_inputs = text2sql_tokenizer(
        input_sequence,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    )
    
    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    
    if device == "cuda":
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
    elif device == "mps":
        encoder_input_ids = encoder_input_ids.to("mps")
        encoder_input_attention_mask = encoder_input_attention_mask.to("mps")
    
    # Generate
    with torch.no_grad():
        model_outputs = text2sql_model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            max_length=256,
            decoder_start_token_id=text2sql_model.config.decoder_start_token_id,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
    
    # Reshape outputs
    model_outputs = model_outputs.view(1, num_return_sequences, model_outputs.shape[1])
    
    # Extract table.column names for fix_fatal_errors_in_natsql
    # This is needed to fix schema errors in the predicted NatSQL
    tc_original = []
    if ranked_data and "db_schema" in ranked_data:
        for table in ranked_data["db_schema"]:
            table_name = table.get("table_name_original", "")
            for col_name in table.get("column_names_original", []):
                tc_original.append(f"{table_name}.{col_name}")
    
    # Decode
    if target_type == "sql":
        sqls = decode_sqls(
            db_path,
            model_outputs,
            [db_id],
            [input_sequence],
            text2sql_tokenizer,
            [[]]
        )
    else:  # natsql
        sqls = decode_natsqls(
            db_path,
            model_outputs,
            [db_id],
            [input_sequence],
            text2sql_tokenizer,
            [tc_original],  # Pass the actual table.column names
            tables_dict
        )
    
    return sqls[0] if sqls else "SELECT * FROM table"

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global device
    try:
        device = load_models(model_scale="base", target_type="natsql")
        print("API server ready!")
    except Exception as e:
        print(f"Warning: Could not load models on startup: {e}")
        print("Models will be loaded on first request")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = (
        schema_classifier_model is not None and
        text2sql_model is not None
    )
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded,
        device=device or "unknown"
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer_sql(request: InferenceRequest):
    """Generate SQL from natural language question"""
    global schema_classifier_model, text2sql_model, all_db_infos, device
    
    # Check if models are loaded
    if schema_classifier_model is None or text2sql_model is None:
        try:
            device = load_models(model_scale="base", target_type=request.target_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    
    try:
        # Set paths
        db_path = "./database"
        tables_path = "./data/spider/tables.json"
        
        # Load tables if not loaded
        if all_db_infos is None:
            with open(tables_path, 'r', encoding='utf-8') as f:
                all_db_infos = json.load(f)
        
        # Step 1: Preprocess
        preprocessed_data = preprocess_single_question(
            request.question,
            request.db_id,
            all_db_infos,
            db_path,
            request.target_type
        )
        
        # Step 2: Classify schema items
        preprocessed_data = classify_schema_items(
            preprocessed_data,
            use_contents=request.use_contents,
            add_fk_info=request.add_fk_info
        )
        
        # Step 3: Generate input sequence
        input_sequence, ranked_data = generate_input_sequence(
            preprocessed_data,
            target_type=request.target_type,
            use_contents=request.use_contents,
            add_fk_info=request.add_fk_info,
            topk_table_num=request.topk_table_num,
            topk_column_num=request.topk_column_num
        )
        
        # Step 4: Generate SQL
        sql = generate_sql(
            input_sequence,
            request.db_id,
            db_path,
            target_type=request.target_type,
            num_beams=request.num_beams,
            num_return_sequences=request.num_return_sequences,
            ranked_data=ranked_data
        )
        
        return InferenceResponse(
            sql=sql,
            input_sequence=input_sequence,
            execution_success=sql != "sql placeholder"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RESDSQL Text-to-SQL API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/infer": "POST - Generate SQL from natural language",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
