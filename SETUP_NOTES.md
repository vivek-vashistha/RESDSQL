# RESDSQL Environment Setup Notes

## Setup Completed Successfully ✅

The RESDSQL environment has been set up on your MacBook Pro 2021 (ARM64) with the following:

1. ✅ Conda environment `resdsql` created with Python 3.8.11 (3.8.5 was not available for ARM64)
2. ✅ PyTorch 2.4.1 installed (Mac ARM64 version with MPS support)
3. ✅ Most dependencies from requirements.txt installed:
   - editdistance (upgraded to 0.8.1 for ARM64 compatibility)
   - protobuf==3.19.0
   - func_timeout==4.3.5
   - nltk==3.7
   - numpy==1.22.3
   - rapidfuzz==2.0.11
   - scikit_learn==1.2.1
   - sql_metadata==2.6.0
   - sqlparse==0.4.2
   - tokenizers==0.11.6
   - tqdm==4.63.0
   - transformers==4.28.1
   - tensorboard==2.8.0
   - sentencepiece==0.1.99
4. ✅ NLTK punkt data downloaded
5. ✅ Required folders created (eval_results, models, tensorboard_log, third_party, predictions)
6. ✅ Evaluation scripts cloned (spider and test_suite)

## ⚠️ Known Issue: Spacy Installation

**Spacy and the spacy model could not be installed** due to compilation issues with C++ dependencies (cymem, murmurhash) on ARM64 Mac. The error indicates that the C++ standard library headers cannot be found during compilation.

### Solutions to Fix Spacy Installation:

#### Option 1: Install Xcode Command Line Tools (Recommended)
```bash
xcode-select --install
```
After installation, try installing spacy again:
```bash
conda activate resdsql
pip install spacy==2.2.3
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

#### Option 2: Use Conda to Install Spacy
```bash
conda activate resdsql
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```

#### Option 3: Install Pre-built Wheels (if available)
Check if there are pre-built wheels for ARM64:
```bash
conda activate resdsql
pip install --only-binary :all: spacy==2.2.3
```

## Activating the Environment

To activate the environment:
```bash
conda activate resdsql
```

## Next Steps

1. Fix spacy installation using one of the options above
2. Download the data and database files as mentioned in the README
3. Download the model checkpoints as mentioned in the README
4. You're ready to run inference or training!

## Notes

- Python version: 3.8.11 (instead of 3.8.5 as specified in README, due to ARM64 availability)
- PyTorch version: 2.4.1 (Mac ARM64 version, supports MPS for GPU acceleration)
- Editdistance: Upgraded to 0.8.1 (0.6.2 had compilation issues on ARM64)
