# RESDSQL Inference Guide for Spider Dataset

## System Requirements Analysis

Based on your MacBook Pro 2021 (ARM64, 16GB RAM, 10 vCPUs):

### Model Size Recommendations:

| Model | Parameters | Approx. Size | RAM Usage | Recommendation for 16GB RAM |
|-------|-----------|--------------|-----------|----------------------------|
| **T5-Base** | ~220M | ~850MB | ~2-3GB | ✅ **RECOMMENDED** - Best fit |
| **T5-Large** | ~770M | ~3GB | ~5-6GB | ⚠️ **MARGINAL** - May work but tight |
| **T5-3B** | ~3B | ~11-12GB | ~14-16GB | ❌ **NOT RECOMMENDED** - Too large for 16GB |

**Recommendation: Use T5-Base model** for your system configuration.

## Step-by-Step Inference Setup

### Step 1: Fix Spacy Installation (Required)

First, ensure spacy is installed (needed for preprocessing):

```bash
conda activate resdsql

# Option 1: Install Xcode Command Line Tools first (if not already installed)
xcode-select --install

# Then install spacy
pip install spacy==2.2.3
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

### Step 2: Download Data and Databases

Download the Spider dataset and databases:

1. **Spider Data**: [Download Link](https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing)
2. **Databases**: [Download Link](https://drive.google.com/file/d/1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo/view?usp=share_link)

After downloading, extract them:

```bash
cd /Users/vivekvashistha/Projects/Clients/Turing/Projects/RESDSQL
unzip data.zip
unzip database.zip
```

This will create:
- `./data/spider/` directory with Spider dataset
- `./database/` directory with SQLite databases

### Step 3: Download Model Checkpoints

For Spider inference, you need **TWO** checkpoints:

#### A. Cross-encoder Checkpoint (Schema Item Classifier)

**For NatSQL version (recommended):**
- **Name**: `text2natsql_schema_item_classifier`
- **Google Drive**: [Link](https://drive.google.com/file/d/1UWNj1ZADfKa1G5I4gBYCJeEQO6piMg4G/view?usp=share_link)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/15eGPMSx7K8oLV4hkjnCzaw) (pwd: 18w8)

**For SQL version:**
- **Name**: `text2sql_schema_item_classifier`
- **Google Drive**: [Link](https://drive.google.com/file/d/1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo/view?usp=share_link)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1trSi8OBOcPo5NkZb_o-T4g) (pwd: dr62)

#### B. T5 Model Checkpoint (Choose ONE based on your RAM)

**For T5-Base (RECOMMENDED for 16GB RAM):**

**NatSQL version:**
- **Name**: `text2natsql-t5-base`
- **Path in models folder**: `./models/text2natsql-t5-base/checkpoint-14352`
- **Google Drive**: [Link](https://drive.google.com/file/d/1QyfSfHHrxfIM5X9gKUYNr_0ZRVvb1suV/view?usp=share_link)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1XegaZFvXuZ_jf3P-9YPQCQ) (pwd: pyxf)

**SQL version:**
- **Name**: `text2sql-t5-base`
- **Path in models folder**: `./models/text2sql-t5-base/checkpoint-39312`
- **Google Drive**: [Link](https://drive.google.com/file/d/1lqZ81f_fSZtg6BRcRw1-Ol-RJCcKRsmH/view?usp=sharing)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1-6H7zStq0WCJHTjDuVspoQ) (pwd: wuek)

**For T5-Large (if you want to try, but may be tight on 16GB RAM):**

**NatSQL version:**
- **Name**: `text2natsql-t5-large`
- **Path**: `./models/text2natsql-t5-large/checkpoint-21216`
- **Google Drive**: [Link](https://drive.google.com/file/d/1ZwFsH24_qKC3xwYdedPi6T_8argguWHe/view?usp=sharing)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/18H8lgnv9gfXmUo_oO_CdOA) (pwd: 7iyq)

**SQL version:**
- **Name**: `text2sql-t5-large`
- **Path**: `./models/text2sql-t5-large/checkpoint-30576`
- **Google Drive**: [Link](https://drive.google.com/file/d/1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e/view?usp=sharing)
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1Mwg0OZZ48APEq9jPvQQNtw) (pwd: q58k)

#### Organize Checkpoints

After downloading, place them in the `models` folder with the correct structure:

```bash
models/
├── text2natsql_schema_item_classifier/    # Cross-encoder for NatSQL
│   └── (checkpoint files)
├── text2natsql-t5-base/                    # T5-Base for NatSQL
│   └── checkpoint-14352/
│       └── (checkpoint files)
└── text2sql_schema_item_classifier/        # Cross-encoder for SQL (if using SQL version)
    └── (checkpoint files)
```

### Step 4: Modify Inference Script for Mac

The inference scripts use `device="0"` which assumes a GPU. For Mac, you need to modify this.

**Option 1: Modify the script temporarily**

Edit `scripts/inference/infer_text2natsql.sh` and change:
```bash
device="0"
```
to:
```bash
device="cpu"  # or "mps" if you want to use Apple's Metal Performance Shaders
```

**Option 2: Use environment variable (Recommended)**

You can override the device by setting an environment variable before running:

```bash
export CUDA_VISIBLE_DEVICES=""  # Forces CPU usage
```

### Step 5: Run Inference

#### For NatSQL version (Recommended - Better Performance):

```bash
conda activate resdsql
cd /Users/vivekvashistha/Projects/Clients/Turing/Projects/RESDSQL

# Set device to CPU (or MPS for Apple Silicon GPU)
export CUDA_VISIBLE_DEVICES=""

# Run inference with T5-Base (recommended for 16GB RAM)
sh scripts/inference/infer_text2natsql.sh base spider
```

#### For SQL version:

```bash
conda activate resdsql
cd /Users/vivekvashistha/Projects/Clients/Turing/Projects/RESDSQL

# Set device to CPU
export CUDA_VISIBLE_DEVICES=""

# Run inference with T5-Base
sh scripts/inference/infer_text2sql.sh base spider
```

### Step 6: Check Results

The predicted SQL queries will be saved in:
```
./predictions/Spider-dev/resdsql_base_natsql/pred.sql
```
or
```
./predictions/Spider-dev/resdsql_base/pred.sql
```

## Performance Expectations

Based on the README results:

| Model | Dev EM | Dev EX | Inference Speed (Estimated) |
|-------|--------|--------|----------------------------|
| RESDSQL-Base+NatSQL | 74.1% | 80.2% | Fastest (batch_size=16) |
| RESDSQL-Base | 71.7% | 77.9% | Fastest (batch_size=16) |
| RESDSQL-Large+NatSQL | 76.7% | 81.9% | Medium (batch_size=8) |
| RESDSQL-3B+NatSQL | 80.5% | 84.1% | Slowest (batch_size=6) |

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

1. **Reduce batch size**: Edit the inference script and reduce `text2natsql_model_bs` or `text2sql_model_bs`
2. **Use CPU only**: Ensure `device="cpu"` is set
3. **Close other applications**: Free up RAM before running inference

### Slow Inference

- Inference on CPU will be slower than GPU
- T5-Base should complete Spider dev set in reasonable time (30-60 minutes on CPU)
- Consider using MPS (Metal Performance Shaders) if available: `device="mps"`

### Missing Dependencies

If you get import errors, ensure all dependencies are installed:
```bash
conda activate resdsql
pip install -r requirements.txt
```

## Summary Checklist

- [ ] Fix spacy installation
- [ ] Download Spider data and extract to `./data/spider/`
- [ ] Download databases and extract to `./database/`
- [ ] Download cross-encoder checkpoint (`text2natsql_schema_item_classifier` or `text2sql_schema_item_classifier`)
- [ ] Download T5-Base checkpoint (`text2natsql-t5-base` or `text2sql-t5-base`)
- [ ] Place checkpoints in correct `models/` subdirectories
- [ ] Modify device setting in script or use environment variable
- [ ] Run inference script
- [ ] Check results in `predictions/` folder

## Quick Start Command

Once everything is downloaded and set up:

```bash
conda activate resdsql
cd /Users/vivekvashistha/Projects/Clients/Turing/Projects/RESDSQL
export CUDA_VISIBLE_DEVICES=""
sh scripts/inference/infer_text2natsql.sh base spider
```

This will run RESDSQL-Base+NatSQL on Spider's dev set, which should work well on your 16GB RAM MacBook Pro.
