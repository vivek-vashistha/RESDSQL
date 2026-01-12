# FastAPI Implementation Analysis

## Comparison: FastAPI vs Original Pipeline

### Original Pipeline Steps (from `infer_text2natsql.sh`)

1. **Preprocessing** (`preprocessing.py --mode "test"`)
   - Processes JSON dataset file
   - Extracts database schemas
   - Gets database contents
   - Builds preprocessed data structure

2. **Schema Item Classification** (`schema_item_classifier.py --mode "test"`)
   - Loads preprocessed data
   - Runs classifier model
   - Adds `table_probs` and `column_probs` to data
   - Outputs JSON with probabilities

3. **Data Generation** (`text2sql_data_generator.py`)
   - Takes data with probabilities
   - Ranks tables/columns by probability
   - Generates input sequence using `prepare_input_and_output()`
   - Outputs final dataset for text2sql model

4. **SQL Generation** (`text2sql.py --mode "eval"`)
   - Loads final dataset
   - Runs T5 model inference
   - Decodes SQL/NatSQL using `decode_sqls()` or `decode_natsqls()`

---

## FastAPI Implementation Analysis

### ✅ **Component 1: Preprocessing** - PARTIALLY RE-IMPLEMENTED

**FastAPI Function:** `preprocess_single_question()` (lines 243-335)

**What it REUSES:**
- ✅ `get_db_schemas()` from `preprocessing.py` (imported)
- ✅ `get_db_contents()` from `preprocessing.py` (imported)

**What it RE-IMPLEMENTS:**
- ❌ The main preprocessing logic (building `preprocessed_data` structure)
- ❌ Table/column schema building loop
- ❌ Data structure creation

**Original Code Path:**
- `preprocessing.py` → `main()` function (lines 270-410) processes entire dataset
- Uses same helper functions but in a batch processing context

**Verdict:** The FastAPI re-implements the preprocessing logic but uses the same helper functions. This is reasonable for single-question processing, but the logic should match exactly.

---

### ✅ **Component 2: Schema Item Classification** - PARTIALLY RE-IMPLEMENTED

**FastAPI Function:** `classify_schema_items()` (lines 337-406)

**What it REUSES:**
- ✅ `prepare_batch_inputs_and_labels()` from `schema_item_classifier.py` (imported)
- ✅ Model architecture (`MyClassifier` from `utils.classifier_model`)

**What it RE-IMPLEMENTS:**
- ❌ Model loading logic (but similar to original)
- ❌ Batch preparation (creates custom batch format)
- ❌ Probability extraction logic
- ❌ The entire test/inference flow

**Original Code Path:**
- `schema_item_classifier.py` → `_test()` function (lines 374-458)
- Uses `ColumnAndTableClassifierDataset` to load preprocessed data
- Processes batches through DataLoader

**Verdict:** The FastAPI re-implements the classification inference logic. It uses the same helper function `prepare_batch_inputs_and_labels()`, but creates its own batch format instead of using the dataset class. This could lead to inconsistencies.

---

### ✅ **Component 3: Data Generation** - PROPERLY REUSED

**FastAPI Function:** `generate_input_sequence()` (lines 408-469)

**What it REUSES:**
- ✅ `prepare_input_and_output()` from `text2sql_data_generator.py` (imported) - **PROPERLY REUSED**

**What it RE-IMPLEMENTS:**
- ⚠️ Table/column ranking logic (but this is straightforward and matches original logic)
- ⚠️ Creates a custom `Opt` class to pass to `prepare_input_and_output()`

**Original Code Path:**
- `text2sql_data_generator.py` → `prepare_input_and_output()` function (lines 42-94)
- Called from `generate_train_ranked_dataset()` or `generate_test_ranked_dataset()`

**Verdict:** ✅ **This component is properly reused!** The FastAPI correctly imports and uses `prepare_input_and_output()` from the original module.

---

### ✅ **Component 4: SQL Generation** - PARTIALLY RE-IMPLEMENTED

**FastAPI Function:** `generate_sql()` (lines 471-566)

**What it REUSES:**
- ✅ `decode_sqls()` from `utils.text2sql_decoding_utils` (imported)
- ✅ `decode_natsqls()` from `utils.text2sql_decoding_utils` (imported)
- ✅ Model architecture (T5/MT5 from transformers)

**What it RE-IMPLEMENTS:**
- ❌ Model loading logic (but similar to original)
- ❌ Tokenization logic
- ❌ Model inference/generation logic
- ❌ The entire test/inference flow

**Original Code Path:**
- `text2sql.py` → `_test()` function (lines 227-345)
- Uses `Text2SQLDataset` to load data
- Processes batches through DataLoader
- Uses same decode functions

**Verdict:** The FastAPI re-implements the SQL generation inference logic. It uses the same decode functions, which is good, but the model inference logic is duplicated.

---

## Summary

| Component | Status | Reuses Core Logic? | Risk Level |
|-----------|--------|-------------------|------------|
| **Preprocessing** | Re-implemented | ✅ Helper functions | 🟡 Medium |
| **Schema Classification** | Re-implemented | ✅ Helper function | 🟡 Medium |
| **Data Generation** | ✅ Properly Reused | ✅ Core function | 🟢 Low |
| **SQL Generation** | Re-implemented | ✅ Decode functions | 🟡 Medium |

## Issues Identified

1. **Inconsistency Risk:** The FastAPI re-implements several components instead of reusing the original functions. This could lead to:
   - Different behavior between batch inference and API inference
   - Bugs that exist in one but not the other
   - Maintenance burden (fixes need to be applied in two places)

2. **Missing Dataset Classes:** The FastAPI doesn't use:
   - `ColumnAndTableClassifierDataset` (used in schema classification)
   - `Text2SQLDataset` (used in SQL generation)
   
   Instead, it manually creates batch structures, which might not match exactly.

3. **Code Duplication:** Significant code duplication in:
   - Model loading logic
   - Inference loops
   - Batch preparation

## Recommendations

### Option 1: Refactor to Reuse More Components (Ideal)
- Extract reusable functions from `preprocessing.py`, `schema_item_classifier.py`, and `text2sql.py`
- Create a unified inference function that can handle both single questions and batches
- FastAPI should call these extracted functions

### Option 2: Keep Current Implementation but Add Tests
- Add integration tests comparing FastAPI results with batch inference results
- Document any intentional differences
- Ensure both paths stay in sync

### Option 3: Create a Shared Inference Module
- Create a new `inference_utils.py` module with shared functions
- Both FastAPI and batch scripts use this module
- Reduces duplication and ensures consistency

## Conclusion

The FastAPI **does reuse some components** (helper functions, decode functions, data generation function), but it **re-implements significant portions** of the preprocessing, classification, and SQL generation logic. This is not ideal from a maintenance perspective, but it works for single-question inference.

The most critical component (data generation with `prepare_input_and_output()`) is properly reused, which is good. However, the preprocessing and classification steps being re-implemented could lead to inconsistencies.
