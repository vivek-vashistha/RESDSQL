# Spider2 Quick Reference Card

## 🚀 Quick Start (5 Minutes)

### 1. Setup
```bash
# Convert Spider2 data
python utils/spider2_converter.py \
    --input data/spider2/spider2-lite.jsonl \
    --output_dir data/spider2/converted \
    --dataset_name spider2-lite
```

### 2. Start FastAPI Server
```bash
bash start_api.sh
```

### 3. Test a Question
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many orders are there?",
    "db_id": "E_commerce",
    "target_type": "natsql",
    "spider2_mode": true,
    "tables_path": "./data/spider2/converted/spider2-lite_tables.json"
  }'
```

### 4. Evaluate (SQLite Only)
```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl data/spider2/spider2-lite.jsonl \
  --gold_dir /path/to/Spider2/spider2-lite/evaluation_suite/gold \
  --output results.json
```

---

## 📁 Required Files

| File | Location | Purpose |
|------|----------|---------|
| `spider2-lite.jsonl` | `data/spider2/` | Questions dataset |
| `spider2-lite_tables.json` | `data/spider2/converted/` | Schema information (REQUIRED) |
| SQLite databases | `database/{db_id}/` | Database files |
| Gold results | `/path/to/Spider2/.../gold/` | Evaluation results |

---

## 🔧 Common Commands

### Data Conversion
```bash
python utils/spider2_converter.py \
    --input data/spider2/spider2-lite.jsonl \
    --output_dir data/spider2/converted \
    --dataset_name spider2-lite
```

### Batch Inference
```bash
bash scripts/inference/infer_text2natsql_spider2.sh base lite
```

### FastAPI Evaluation
```bash
python evaluate_spider2_with_api.py \
  --spider2_jsonl data/spider2/spider2-lite.jsonl \
  --gold_dir /path/to/gold \
  --api_timeout 600 \
  --max_retries 3 \
  --output results.json
```

### Batch Evaluation
```bash
python evaluate_spider2.py \
    --predictions predictions/spider2-lite/resdsql_base_natsql/pred.sql \
    --spider2_data data/spider2/spider2-lite.jsonl \
    --output results.json
```

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "FastAPI server not running" | `bash start_api.sh` |
| "Tables file not found" | Run `spider2_converter.py` |
| "Database file not found" | Check `database/{db_id}/{db_id}.sqlite` |
| "API timeout" | Use `--api_timeout 600` |
| "Gold CSV not found" | Verify `--gold_dir` path |

---

## 📊 Workflow Comparison

| Feature | FastAPI Server | Batch Pipeline |
|---------|---------------|----------------|
| **Speed** | 2-5 sec/question | 30-50 sec/question |
| **Use Case** | Interactive, testing | Full dataset |
| **Setup** | Start server | Run script |
| **Output** | Per-request | Batch file |

---

## 📚 Documentation

- **Complete Guide**: `SPIDER2_COMPLETE_GUIDE.md`
- **FastAPI**: `FASTAPI_SPIDER2_README.md`
- **Evaluation**: `EVALUATE_SPIDER2_API_README.md`
- **Quick Start**: `QUICK_START_SPIDER2_API.md`

---

## 🔗 Key Paths

```bash
# Questions
data/spider2/spider2-lite.jsonl

# Schema (REQUIRED)
data/spider2/converted/spider2-lite_tables.json

# Databases
database/{db_id}/{db_id}.sqlite

# Predictions
predictions/spider2-lite/resdsql_base_natsql/pred.sql

# Gold Results
/path/to/Spider2/spider2-lite/evaluation_suite/gold/
```

---

**For detailed information, see `SPIDER2_COMPLETE_GUIDE.md`**
