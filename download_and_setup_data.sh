#!/bin/bash

# ============================================================================
# RESDSQL Data Download and Setup Script
# ============================================================================
# This script downloads and sets up data, databases, and model checkpoints
# for RESDSQL from Google Drive.
#
# Default behavior: Downloads Spider data, databases, and base model
#   (text2natsql-t5-base + text2natsql_schema_item_classifier)
#
# Usage:
#   chmod +x download_and_setup_data.sh
#   ./download_and_setup_data.sh                    # Default: Spider data + base model
#   ./download_and_setup_data.sh --data-only        # Download only data
#   ./download_and_setup_data.sh --databases-only   # Download only databases
#   ./download_and_setup_data.sh --all              # Download everything (all models)
#   ./download_and_setup_data.sh --models           # Download with interactive model selection
#   ./download_and_setup_data.sh --interactive      # Interactive mode for all options
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
}

# Parse command line arguments
# Default: Download data, databases, and base model (non-interactive)
DOWNLOAD_DATA=true
DOWNLOAD_DATABASES=true
DOWNLOAD_MODELS=true
DOWNLOAD_BASE_MODEL_ONLY=true
INTERACTIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=false
            DOWNLOAD_MODELS=false
            INTERACTIVE=false
            shift
            ;;
        --databases-only)
            DOWNLOAD_DATA=false
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=false
            INTERACTIVE=false
            shift
            ;;
        --all)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=true
            DOWNLOAD_BASE_MODEL_ONLY=false
            INTERACTIVE=false
            shift
            ;;
        --models)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=true
            DOWNLOAD_BASE_MODEL_ONLY=false
            INTERACTIVE=true
            shift
            ;;
        --interactive|-i)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=true
            DOWNLOAD_BASE_MODEL_ONLY=false
            INTERACTIVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Default behavior (no options): Downloads Spider data, databases, and base model"
            echo "  (text2natsql-t5-base + text2natsql_schema_item_classifier)"
            echo ""
            echo "Options:"
            echo "  --data-only        Download only data files"
            echo "  --databases-only   Download only database files"
            echo "  --all              Download everything (all models) non-interactively"
            echo "  --models           Download data, databases, and models (interactive model selection)"
            echo "  --interactive, -i  Interactive mode for selecting what to download"
            echo "  --help, -h         Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

print_section "RESDSQL Data Download and Setup Script"

# Show what will be downloaded by default
if [ "$DOWNLOAD_DATA" = true ] && [ "$DOWNLOAD_DATABASES" = true ] && [ "$DOWNLOAD_BASE_MODEL_ONLY" = true ]; then
    print_info "Default mode: Downloading Spider data, databases, and base model"
    print_info "  - Data files (Spider dataset)"
    print_info "  - Database files"
    print_info "  - text2natsql_schema_item_classifier (cross-encoder)"
    print_info "  - text2natsql-t5-base (T5 model)"
    echo ""
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warn "Conda environment not activated. Attempting to activate..."
    if [ -d "$HOME/miniconda3" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
        if conda env list | grep -q "^resdsql "; then
            conda activate resdsql 2>/dev/null || print_warn "Could not activate conda environment. Continuing anyway..."
        fi
    fi
fi

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    print_info "Installing gdown for Google Drive downloads..."
    pip install gdown --quiet
else
    print_info "gdown is already installed."
fi

# Google Drive file IDs
DATA_ID="19tsgBGAxpagULSl9r85IFKIZb4kyBGGu"
DATABASE_ID="1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo"

# Model checkpoints
declare -A MODEL_IDS=(
    ["text2natsql_schema_item_classifier"]="1UWNj1ZADfKa1G5I4gBYCJeEQO6piMg4G"
    ["text2sql_schema_item_classifier"]="1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo"
    ["text2natsql-t5-base"]="1QyfSfHHrxfIM5X9gKUYNr_0ZRVvb1suV"
    ["text2natsql-t5-large"]="1ZwFsH24_qKC3xwYdedPi6T_8argguWHe"
    ["text2sql-t5-base"]="1lqZ81f_fSZtg6BRcRw1-Ol-RJCcKRsmH"
    ["text2sql-t5-large"]="1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e"
    ["text2sql-t5-3b"]="1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11"
)

# Function to download and extract a file
download_and_extract() {
    local file_id=$1
    local output_file=$2
    local extract_dir=$3
    local check_dir=$4
    local description=$5
    
    if [ -d "$check_dir" ] && [ "$(ls -A $check_dir 2>/dev/null)" ]; then
        print_info "$description already exists. Skipping download."
        return 0
    fi
    
    print_info "Downloading $description (this may take a while)..."
    print_info "File ID: $file_id"
    
    if gdown --id "$file_id" -O "$output_file" --fuzzy; then
        print_info "Download completed: $output_file"
        
        if [ -f "$output_file" ]; then
            print_info "Extracting $output_file..."
            if unzip -q "$output_file" -d "$extract_dir" 2>/dev/null || unzip -q "$output_file" 2>/dev/null; then
                print_info "Extraction completed successfully."
                rm -f "$output_file"
                return 0
            else
                print_error "Failed to extract $output_file"
                return 1
            fi
        else
            print_error "Downloaded file not found: $output_file"
            return 1
        fi
    else
        print_error "Failed to download $description"
        print_info "You may need to download manually from Google Drive"
        return 1
    fi
}

# Function to download a model checkpoint
download_model() {
    local model_name=$1
    local model_id=$2
    
    local model_dir="models/$model_name"
    local zip_file="${model_dir}.zip"
    
    if [ -d "$model_dir" ] && [ "$(ls -A $model_dir 2>/dev/null)" ]; then
        print_info "Model $model_name already exists. Skipping download."
        return 0
    fi
    
    print_info "Downloading $model_name..."
    mkdir -p "$model_dir"
    
    if gdown --id "$model_id" -O "$zip_file" --fuzzy; then
        if [ -f "$zip_file" ]; then
            print_info "Extracting $model_name..."
            cd "$model_dir"
            if unzip -q "../$(basename $zip_file)"; then
                print_info "Extraction completed successfully."
                cd "$PROJECT_DIR"
                rm -f "$zip_file"
                return 0
            else
                print_error "Failed to extract $zip_file"
                cd "$PROJECT_DIR"
                return 1
            fi
        fi
    else
        print_error "Failed to download $model_name"
        cd "$PROJECT_DIR"
        return 1
    fi
}

# Interactive mode: Ask what to download (only if --interactive flag is used)
if [ "$INTERACTIVE" = true ] && [ "$DOWNLOAD_BASE_MODEL_ONLY" = false ]; then
    print_section "What would you like to download?"
    echo ""
    echo "  1) Data files only"
    echo "  2) Database files only"
    echo "  3) Data + Database files"
    echo "  4) Data + Database + Model checkpoints (interactive selection)"
    echo "  5) Everything (all models, non-interactive)"
    echo "  6) Cancel"
    echo ""
    read -p "Enter choice (1-6): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=false
            DOWNLOAD_MODELS=false
            ;;
        2)
            DOWNLOAD_DATA=false
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=false
            ;;
        3)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=false
            ;;
        4)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=true
            DOWNLOAD_BASE_MODEL_ONLY=false
            ;;
        5)
            DOWNLOAD_DATA=true
            DOWNLOAD_DATABASES=true
            DOWNLOAD_MODELS=true
            DOWNLOAD_BASE_MODEL_ONLY=false
            INTERACTIVE=false
            ;;
        6)
            print_info "Cancelled."
            exit 0
            ;;
        *)
            print_error "Invalid choice."
            exit 1
            ;;
    esac
fi

# Download data
if [ "$DOWNLOAD_DATA" = true ]; then
    print_section "Downloading Data Files"
    download_and_extract "$DATA_ID" "data.zip" "." "data/spider" "Data files"
fi

# Download databases
if [ "$DOWNLOAD_DATABASES" = true ]; then
    print_section "Downloading Database Files"
    download_and_extract "$DATABASE_ID" "database.zip" "." "database" "Database files"
fi

# Download models
if [ "$DOWNLOAD_MODELS" = true ]; then
    print_section "Downloading Model Checkpoints"
    
    if [ "$DOWNLOAD_BASE_MODEL_ONLY" = true ]; then
        # Default: Download only base model for Spider
        print_info "Downloading base model for Spider (default)..."
        print_info "  - text2natsql_schema_item_classifier (cross-encoder)"
        print_info "  - text2natsql-t5-base (T5 model)"
        download_model "text2natsql_schema_item_classifier" "${MODEL_IDS[text2natsql_schema_item_classifier]}"
        download_model "text2natsql-t5-base" "${MODEL_IDS[text2natsql-t5-base]}"
    elif [ "$INTERACTIVE" = true ]; then
        # Interactive model selection
        print_info "Which cross-encoder do you want to download?"
        echo "  1) text2natsql_schema_item_classifier (recommended for NatSQL)"
        echo "  2) text2sql_schema_item_classifier (for SQL version)"
        echo "  3) Skip cross-encoder"
        read -p "Enter choice (1/2/3): " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                download_model "text2natsql_schema_item_classifier" "${MODEL_IDS[text2natsql_schema_item_classifier]}"
                ;;
            2)
                download_model "text2sql_schema_item_classifier" "${MODEL_IDS[text2sql_schema_item_classifier]}"
                ;;
        esac
        
        print_info "Which T5 model do you want to download?"
        echo "  1) text2natsql-t5-base (recommended for 16GB RAM)"
        echo "  2) text2natsql-t5-large"
        echo "  3) text2sql-t5-base"
        echo "  4) text2sql-t5-large"
        echo "  5) text2sql-t5-3b"
        echo "  6) Skip T5 model"
        read -p "Enter choice (1-6): " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                download_model "text2natsql-t5-base" "${MODEL_IDS[text2natsql-t5-base]}"
                ;;
            2)
                download_model "text2natsql-t5-large" "${MODEL_IDS[text2natsql-t5-large]}"
                ;;
            3)
                download_model "text2sql-t5-base" "${MODEL_IDS[text2sql-t5-base]}"
                ;;
            4)
                download_model "text2sql-t5-large" "${MODEL_IDS[text2sql-t5-large]}"
                ;;
            5)
                download_model "text2sql-t5-3b" "${MODEL_IDS[text2sql-t5-3b]}"
                ;;
        esac
    else
        # Non-interactive: download all models
        print_info "Downloading all available models (non-interactive mode)..."
        for model_name in "${!MODEL_IDS[@]}"; do
            download_model "$model_name" "${MODEL_IDS[$model_name]}"
        done
    fi
fi

# Verification
print_section "Verification"

print_info "Checking downloaded files..."
VERIFICATION_PASSED=true

if [ "$DOWNLOAD_DATA" = true ]; then
    if [ -d "data/spider" ] && [ "$(ls -A data/spider 2>/dev/null)" ]; then
        print_info "✓ Data files found in data/spider"
    else
        print_error "✗ Data files not found or empty"
        VERIFICATION_PASSED=false
    fi
fi

if [ "$DOWNLOAD_DATABASES" = true ]; then
    if [ -d "database" ] && [ "$(ls -A database 2>/dev/null)" ]; then
        print_info "✓ Database files found in database/"
        DB_COUNT=$(find database -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        print_info "  Found $DB_COUNT database directories"
    else
        print_error "✗ Database files not found or empty"
        VERIFICATION_PASSED=false
    fi
fi

if [ "$DOWNLOAD_MODELS" = true ]; then
    MODEL_COUNT=0
    for model_name in "${!MODEL_IDS[@]}"; do
        if [ -d "models/$model_name" ] && [ "$(ls -A models/$model_name 2>/dev/null)" ]; then
            print_info "✓ Model found: $model_name"
            MODEL_COUNT=$((MODEL_COUNT + 1))
        fi
    done
    if [ $MODEL_COUNT -eq 0 ]; then
        print_warn "No models found in models/ directory"
    else
        print_info "  Found $MODEL_COUNT model(s)"
    fi
fi

echo ""
if [ "$VERIFICATION_PASSED" = true ]; then
    print_section "Download and Setup Completed Successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Verify your setup:"
    echo "     ls -lh data/spider/"
    echo "     ls -lh database/"
    echo "     ls -lh models/"
    echo ""
    echo "  2. Run inference:"
    echo "     conda activate resdsql"
    echo "     sh scripts/inference/infer_text2natsql.sh base spider"
else
    print_warn "Some downloads may have failed. Please check the errors above."
    print_info "You can re-run this script to retry failed downloads."
fi

echo ""
