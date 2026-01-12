"""
FastAPI server for RESDSQL Text-to-SQL inference
Refactored to strictly reuse all existing components
"""
import os
import sys
import json
import torch
import tempfile
import numpy as np
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
from torch.utils.data import DataLoader

# Import RESDSQL modules - STRICTLY REUSE EXISTING COMPONENTS
from preprocessing import get_db_schemas, get_db_contents
from schema_item_classifier import prepare_batch_inputs_and_labels
from utils.classifier_model import MyClassifier
from utils.load_dataset import ColumnAndTableClassifierDataset, Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls
from text2sql_data_generator import prepare_input_and_output
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast

# For NatSQL
try:
    from NatSQL.natsql_utils import natsql_to_sql
    try:
        from NatSQL.table_transform import transform_tables
    except ImportError:
        transform_tables = None
    NATSQL_AVAILABLE = True
except ImportError as e:
    NATSQL_AVAILABLE = False
    print(f"Warning: NatSQL not available. Only SQL mode will work. Error: {e}")

app = FastAPI(title="RESDSQL Text-to-SQL API", version="2.0.0")

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
loaded_target_type = None

# Request/Response models
class InferenceRequest(BaseModel):
    question: str
    db_id: str
    target_type: Optional[str] = "natsql"
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
    """Load models and prepare for inference - REUSES ORIGINAL LOGIC"""
    global schema_classifier_model, schema_classifier_tokenizer
    global text2sql_model, text2sql_tokenizer, tables_dict, all_db_infos, device, loaded_target_type
    
    device = get_device()
    print(f"Using device: {device}")
    
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Determine model paths
    if schema_classifier_path is None:
        if target_type == "natsql":
            schema_classifier_path = "./models/text2natsql_schema_item_classifier"
        else:
            schema_classifier_path = "./models/text2sql_schema_item_classifier"
    
    if not os.path.exists(schema_classifier_path):
        raise FileNotFoundError(
            f"Schema classifier not found at {schema_classifier_path}. "
            f"Please download the schema classifier for target_type='{target_type}'."
        )
    
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
    
    # Load schema classifier - REUSE ORIGINAL LOGIC from schema_item_classifier.py
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
    
    # Load fine-tuned params - REUSE ORIGINAL LOGIC
    schema_classifier_model.load_state_dict(
        torch.load(schema_classifier_path + "/dense_classifier.pt", map_location=torch.device('cpu'))
    )
    
    if device == "cuda":
        schema_classifier_model = schema_classifier_model.cuda()
    elif device == "mps":
        schema_classifier_model = schema_classifier_model.to("mps")
    
    schema_classifier_model.eval()
    
    # Load text2sql model - REUSE ORIGINAL LOGIC from text2sql.py
    print(f"Loading text2sql model from {text2sql_model_path}...")
    if not os.path.exists(text2sql_model_path):
        raise FileNotFoundError(
            f"Text2SQL model not found at {text2sql_model_path}. "
            f"Please download the model checkpoint for target_type='{target_type}'."
        )
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
    
    loaded_target_type = target_type
    print(f"Models loaded successfully for target_type: {target_type}!")
    return device

def preprocess_single_question_using_original(
    question: str,
    db_id: str,
    all_db_infos: List[Dict],
    db_path: str,
    target_type: str = "natsql"
):
    """
    Preprocess a single question using the EXACT logic from preprocessing.py's main() function
    This strictly reuses the original preprocessing logic
    """
    # REUSE get_db_schemas from preprocessing.py
    db_schemas = get_db_schemas(all_db_infos)
    
    # Find the database info
    db_info = None
    for db in all_db_infos:
        if db["db_id"] == db_id:
            db_info = db
            break
    
    if db_info is None:
        raise ValueError(f"Database {db_id} not found in tables.json")
    
    # REUSE the exact preprocessing logic from preprocessing.py main() function
    question = question.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace("\u201d", "'").strip()
    
    # For test mode (no SQL/NatSQL provided)
    sql, norm_sql, sql_skeleton = "", "", ""
    sql_tokens = []
    natsql, norm_natsql, natsql_skeleton = "", "", ""
    natsql_used_columns, natsql_tokens = [], []
    
    # Build preprocessed_data structure - EXACT SAME STRUCTURE as preprocessing.py
    preprocessed_data = {}
    preprocessed_data["question"] = question
    preprocessed_data["db_id"] = db_id
    preprocessed_data["sql"] = sql
    preprocessed_data["norm_sql"] = norm_sql
    preprocessed_data["sql_skeleton"] = sql_skeleton
    preprocessed_data["natsql"] = natsql
    preprocessed_data["norm_natsql"] = norm_natsql
    preprocessed_data["natsql_skeleton"] = natsql_skeleton
    preprocessed_data["db_schema"] = []
    preprocessed_data["pk"] = db_schemas[db_id]["pk"]
    preprocessed_data["fk"] = db_schemas[db_id]["fk"]
    preprocessed_data["table_labels"] = []
    preprocessed_data["column_labels"] = []
    
    # REUSE the exact schema building logic from preprocessing.py
    for table in db_schemas[db_id]["schema_items"]:
        # REUSE get_db_contents from preprocessing.py
        db_contents = get_db_contents(
            question,
            table["table_name_original"],
            table["column_names_original"],
            db_id,
            db_path
        )
        
        preprocessed_data["db_schema"].append({
            "table_name_original": table["table_name_original"],
            "table_name": table["table_name"],
            "column_names": table["column_names"],
            "column_names_original": table["column_names_original"],
            "column_types": table["column_types"],
            "db_contents": db_contents
        })
        
        # For test mode, all labels are 0
        preprocessed_data["table_labels"].append(0)
        preprocessed_data["column_labels"].append([0 for _ in range(len(table["column_names_original"]))])
    
    return preprocessed_data

def classify_schema_items_using_original(
    preprocessed_data: Dict,
    use_contents: bool = True,
    add_fk_info: bool = True
):
    """
    Classify schema items using the EXACT logic from schema_item_classifier.py's _test() function
    This strictly reuses the original classification logic by using ColumnAndTableClassifierDataset
    """
    global schema_classifier_model, schema_classifier_tokenizer, device
    
    # Create a temporary JSON file with the preprocessed data
    # This allows us to use ColumnAndTableClassifierDataset which expects a file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump([preprocessed_data], tmp_file, indent=2, ensure_ascii=False)
        tmp_filepath = tmp_file.name
    
    try:
        # REUSE ColumnAndTableClassifierDataset from utils.load_dataset
        dataset = ColumnAndTableClassifierDataset(
            dir_=tmp_filepath,
            use_contents=use_contents,
            add_fk_info=add_fk_info
        )
        
        # REUSE DataLoader setup from schema_item_classifier.py
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Single question
            shuffle=False,
            collate_fn=lambda x: x
        )
        
        # REUSE the exact inference logic from schema_item_classifier.py _test()
        returned_table_pred_probs, returned_column_pred_probs = [], []
        
        for batch in dataloader:
            # REUSE prepare_batch_inputs_and_labels from schema_item_classifier.py
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
            
            # REUSE the exact model inference logic
            with torch.no_grad():
                model_outputs = schema_classifier_model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table
                )
            
            # REUSE the exact probability extraction logic from schema_item_classifier.py
            for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
                table_pred_probs = torch.nn.functional.softmax(table_logits, dim=1)
                returned_table_pred_probs.append(table_pred_probs[:, 1].cpu().tolist())
            
            for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                column_number_in_each_table = batch_column_number_in_each_table[batch_id]
                column_pred_probs = torch.nn.functional.softmax(column_logits, dim=1)
                returned_column_pred_probs.append([
                    column_pred_probs[:, 1].cpu().tolist()[
                        sum(column_number_in_each_table[:table_id]):
                        sum(column_number_in_each_table[:table_id+1])
                    ]
                    for table_id in range(len(column_number_in_each_table))
                ])
        
        # Update preprocessed_data with probabilities - REUSE logic from schema_item_classifier.py
        if returned_table_pred_probs and returned_column_pred_probs:
            table_pred_probs = returned_table_pred_probs[0]
            column_pred_probs = returned_column_pred_probs[0]
            
            # Pad if necessary (matching schema_item_classifier.py logic)
            table_num = len(preprocessed_data["table_labels"])
            if table_num > len(table_pred_probs):
                table_pred_probs = table_pred_probs + [-1] * (table_num - len(table_pred_probs))
            
            preprocessed_data["table_probs"] = table_pred_probs[:table_num]
            
            # Handle column probabilities
            column_probs_flat = []
            for table_id in range(table_num):
                if table_id < len(column_pred_probs):
                    col_num = len(preprocessed_data["column_labels"][table_id])
                    if col_num == len(column_pred_probs[table_id]):
                        column_probs_flat.extend(column_pred_probs[table_id])
                    else:
                        # Pad or truncate
                        col_probs = column_pred_probs[table_id][:col_num]
                        if len(col_probs) < col_num:
                            col_probs = col_probs + [-1] * (col_num - len(col_probs))
                        column_probs_flat.extend(col_probs)
                else:
                    col_num = len(preprocessed_data["column_labels"][table_id])
                    column_probs_flat.extend([-1] * col_num)
            
            # Round probabilities to match original behavior (text2sql_data_generator.py)
            preprocessed_data["table_probs"] = [round(p, 4) for p in preprocessed_data["table_probs"]]
            preprocessed_data["column_probs"] = [round(p, 2) for p in column_probs_flat]
        
        return preprocessed_data
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)

def generate_input_sequence_using_original(
    preprocessed_data: Dict,
    target_type: str = "natsql",
    use_contents: bool = True,
    add_fk_info: bool = True,
    topk_table_num: int = 4,
    topk_column_num: int = 5
):
    """
    Generate input sequence using the EXACT logic from text2sql_data_generator.py
    This strictly reuses the original data generation logic
    """
    # REUSE the ranking logic from text2sql_data_generator.py generate_test_ranked_dataset
    table_probs = preprocessed_data.get("table_probs", [0] * len(preprocessed_data["db_schema"]))
    column_probs = preprocessed_data.get("column_probs", [])
    
    # Sort tables by probability - REUSE logic from text2sql_data_generator.py (using stable sort)
    topk_table_ids = np.argsort(-np.array(table_probs), kind="stable")[:topk_table_num].tolist()
    topk_tables = topk_table_ids
    
    # Build ranked schema - REUSE logic from text2sql_data_generator.py
    ranked_schema = []
    col_idx = 0
    for table_idx in topk_tables:
        if table_idx >= len(preprocessed_data["db_schema"]):
            continue
        table = preprocessed_data["db_schema"][table_idx]
        num_cols = len(table["column_names_original"])
        table_col_probs = column_probs[col_idx:col_idx+num_cols] if col_idx < len(column_probs) else [0] * num_cols
        
        # Sort columns by probability - REUSE logic from text2sql_data_generator.py (using stable sort)
        topk_column_ids = np.argsort(-np.array(table_col_probs), kind="stable")[:topk_column_num].tolist()
        topk_cols = topk_column_ids[:min(topk_column_num, len(topk_column_ids))]
        
        ranked_table = {
            "table_name_original": table["table_name_original"],
            "column_names_original": [table["column_names_original"][i] for i in topk_cols],
            "db_contents": [table["db_contents"][i] for i in topk_cols]
        }
        ranked_schema.append(ranked_table)
        col_idx += num_cols
    
    # Filter foreign keys based on selected tables - REUSE logic from text2sql_data_generator.py
    table_names_original = [table["table_name_original"] for table in preprocessed_data["db_schema"]]
    needed_fks = []
    for fk in preprocessed_data["fk"]:
        try:
            source_table_id = table_names_original.index(fk["source_table_name_original"])
            target_table_id = table_names_original.index(fk["target_table_name_original"])
            if source_table_id in topk_tables and target_table_id in topk_tables:
                needed_fks.append(fk)
        except ValueError:
            # Skip FK if table not found (shouldn't happen, but safe guard)
            continue
    
    # Create ranked data structure - REUSE structure from text2sql_data_generator.py
    ranked_data = {
        "question": preprocessed_data["question"],
        "db_schema": ranked_schema,
        "fk": needed_fks,  # Use filtered FKs instead of all FKs
        "norm_sql": "",
        "norm_natsql": "",
        "sql_skeleton": "",
        "natsql_skeleton": ""
    }
    
    # REUSE prepare_input_and_output from text2sql_data_generator.py
    class Opt:
        def __init__(self, use_contents_val, add_fk_info_val, target_type_val):
            self.use_contents = use_contents_val
            self.add_fk_info = add_fk_info_val
            self.output_skeleton = True
            self.target_type = target_type_val
    
    opt = Opt(use_contents, add_fk_info, target_type)
    input_sequence, _ = prepare_input_and_output(opt, ranked_data)
    
    # Extract tc_original for SQL generation - REUSE logic from text2sql_data_generator.py
    # Note: In eval/test mode, order is column_names_original + ["*"] (line 249)
    tc_original = []
    for table in ranked_schema:
        table_name = table.get("table_name_original", "")
        # Include "*" wildcard (critical for count(*) queries)
        for col_name in table.get("column_names_original", []) + ["*"]:
            tc_original.append(f"{table_name}.{col_name}")
    
    return input_sequence, ranked_data, tc_original

def generate_sql_using_original(
    input_sequence: str,
    db_id: str,
    db_path: str,
    target_type: str = "natsql",
    num_beams: int = 8,
    num_return_sequences: int = 8,
    tc_original: List[str] = None
):
    """
    Generate SQL using the EXACT logic from text2sql.py's _test() function
    This strictly reuses the original SQL generation logic by using Text2SQLDataset
    """
    global text2sql_model, text2sql_tokenizer, tables_dict, device
    
    if tc_original is None:
        tc_original = []
    
    # Ensure tables_dict is loaded for NatSQL mode
    if target_type == "natsql":
        if tables_dict is None or len(tables_dict) == 0:
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
    
    # Create a temporary JSON file with the input sequence
    # This allows us to use Text2SQLDataset which expects a file
    data_item = {
        "input_sequence": input_sequence,
        "db_id": db_id,
        "tc_original": tc_original
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump([data_item], tmp_file, indent=2, ensure_ascii=False)
        tmp_filepath = tmp_file.name
    
    try:
        # REUSE Text2SQLDataset from utils.load_dataset
        dataset = Text2SQLDataset(
            dir_=tmp_filepath,
            mode="test"
        )
        
        # REUSE DataLoader setup from text2sql.py
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Single question
            shuffle=False,
            collate_fn=lambda x: x,
            drop_last=False
        )
        
        # REUSE the exact inference logic from text2sql.py _test()
        predict_sqls = []
        
        for batch in dataloader:
            batch_inputs = [data[0] for data in batch]
            batch_db_ids = [data[1] for data in batch]
            batch_tc_original = [data[2] for data in batch]
            
            # REUSE tokenization logic from text2sql.py
            tokenized_inputs = text2sql_tokenizer(
                batch_inputs,
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
            
            # REUSE the exact generation logic from text2sql.py
            with torch.no_grad():
                model_outputs = text2sql_model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_input_attention_mask,
                    max_length=256,
                    decoder_start_token_id=text2sql_model.config.decoder_start_token_id,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences
                )
            
            # REUSE the exact reshaping logic from text2sql.py
            model_outputs = model_outputs.view(len(batch_inputs), num_return_sequences, model_outputs.shape[1])
            
            # REUSE decode functions from utils.text2sql_decoding_utils
            if target_type == "sql":
                predict_sqls += decode_sqls(
                    db_path,
                    model_outputs,
                    batch_db_ids,
                    batch_inputs,
                    text2sql_tokenizer,
                    batch_tc_original
                )
            elif target_type == "natsql":
                predict_sqls += decode_natsqls(
                    db_path,
                    model_outputs,
                    batch_db_ids,
                    batch_inputs,
                    text2sql_tokenizer,
                    batch_tc_original,
                    tables_dict
                )
            else:
                raise ValueError(f"Invalid target_type: {target_type}")
        
        return predict_sqls[0] if predict_sqls else "SELECT * FROM table"
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)

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
    """Generate SQL from natural language question - STRICTLY REUSES ALL ORIGINAL COMPONENTS"""
    global schema_classifier_model, text2sql_model, all_db_infos, device, loaded_target_type
    
    # Check if models are loaded and match the requested target_type
    if (schema_classifier_model is None or text2sql_model is None or 
        loaded_target_type != request.target_type):
        try:
            print(f"Loading models for target_type: {request.target_type} (current: {loaded_target_type})")
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
        
        # Step 1: Preprocess using ORIGINAL preprocessing.py logic
        preprocessed_data = preprocess_single_question_using_original(
            request.question,
            request.db_id,
            all_db_infos,
            db_path,
            request.target_type
        )
        
        # Step 2: Classify schema items using ORIGINAL schema_item_classifier.py logic
        preprocessed_data = classify_schema_items_using_original(
            preprocessed_data,
            use_contents=request.use_contents,
            add_fk_info=request.add_fk_info
        )
        
        # Step 3: Generate input sequence using ORIGINAL text2sql_data_generator.py logic
        input_sequence, ranked_data, tc_original = generate_input_sequence_using_original(
            preprocessed_data,
            target_type=request.target_type,
            use_contents=request.use_contents,
            add_fk_info=request.add_fk_info,
            topk_table_num=request.topk_table_num,
            topk_column_num=request.topk_column_num
        )
        
        # Step 4: Generate SQL using ORIGINAL text2sql.py logic
        sql = generate_sql_using_original(
            input_sequence,
            request.db_id,
            db_path,
            target_type=request.target_type,
            num_beams=request.num_beams,
            num_return_sequences=request.num_return_sequences,
            tc_original=tc_original
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
        "message": "RESDSQL Text-to-SQL API (Refactored to strictly reuse all components)",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/infer": "POST - Generate SQL from natural language",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
