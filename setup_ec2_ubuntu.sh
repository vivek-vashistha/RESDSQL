#!/bin/bash

# ============================================================================
# RESDSQL Setup Script for EC2 Ubuntu (g4dn.xlarge with GPU)
# ============================================================================
# This script sets up everything needed to run RESDSQL on EC2 Ubuntu instance
#
# IMPORTANT NOTES:
# 1. For EC2 g4dn instances, we STRONGLY recommend using AWS Deep Learning AMI
#    (DLAMI) which already includes NVIDIA drivers and CUDA toolkit.
#    This will save significant setup time and avoid compatibility issues.
#    Deep Learning AMI: https://aws.amazon.com/machine-learning/amis/
#
# 2. If using a standard Ubuntu AMI, you may need to manually install:
#    - NVIDIA drivers (sudo ubuntu-drivers autoinstall && sudo reboot)
#    - CUDA toolkit (complex, better to use DLAMI)
#
# 3. This script will:
#    - Install system dependencies
#    - Install Miniconda
#    - Create conda environment with Python 3.8.5
#    - Install PyTorch 1.11.0 with CUDA 11.3 support
#    - Install all Python packages
#    - Set up project directories and clone evaluation scripts
#    - OPTIONALLY download data, databases, and models from Google Drive
#
# 4. After running this script:
#    - If you skipped downloads, download data/databases/models manually (see README.md)
#    - If drivers were installed, reboot: sudo reboot
#
# Usage:
#   chmod +x setup_ec2_ubuntu.sh
#   ./setup_ec2_ubuntu.sh
# ============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "RESDSQL EC2 Ubuntu Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running as root (we don't want that for conda)
if [ "$EUID" -eq 0 ]; then 
    print_error "Please do not run this script as root. Run as a regular user."
    exit 1
fi

# Update system packages
print_info "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    unzip \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# Check and configure NVIDIA drivers and CUDA toolkit
print_info "Checking NVIDIA drivers and CUDA toolkit..."

# Check if NVIDIA drivers are already installed
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA drivers already installed."
    nvidia-smi
else
    print_warn "NVIDIA drivers not found."
    print_info "For EC2 g4dn instances, we STRONGLY recommend using AWS Deep Learning AMI which includes drivers."
    print_info "If you're using a standard Ubuntu AMI, you can install drivers manually:"
    print_info "  sudo apt-get install -y ubuntu-drivers-common"
    print_info "  sudo ubuntu-drivers autoinstall"
    print_info "  sudo reboot"
    print_warn "Skipping automatic driver installation. Please install manually or use Deep Learning AMI."
fi

# Check for CUDA installation
CUDA_FOUND=false
CUDA_PATH=""

# Check for CUDA 11.3 specifically
if [ -d "/usr/local/cuda-11.3" ]; then
    CUDA_PATH="/usr/local/cuda-11.3"
    CUDA_FOUND=true
    print_info "Found CUDA 11.3 installation in $CUDA_PATH"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
    CUDA_FOUND=true
    print_info "Found CUDA installation in $CUDA_PATH"
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_info "CUDA version: $CUDA_VERSION"
    fi
elif command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_info "CUDA already installed: version $CUDA_VERSION"
    CUDA_FOUND=true
    # Try to find CUDA path
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
else
    print_warn "CUDA not found in standard locations."
    print_info "For EC2 instances, we STRONGLY recommend using AWS Deep Learning AMI (DLAMI) which includes CUDA."
    print_info "If you need to install CUDA manually, see: https://developer.nvidia.com/cuda-11.3.0-download-archive"
    print_warn "PyTorch will still install, but GPU support may not work without CUDA."
fi

# Set CUDA paths if CUDA is found
if [ "$CUDA_FOUND" = true ] && [ -n "$CUDA_PATH" ]; then
    if ! grep -q "$CUDA_PATH" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA" >> ~/.bashrc
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    fi
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
fi

# Install Miniconda
print_info "Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="$HOME/miniconda3_installer.sh"
    
    wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm "$MINICONDA_INSTALLER"
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash
    source ~/.bashrc
    
    print_info "Miniconda installed successfully."
else
    print_info "Miniconda already installed."
fi

# Add conda to PATH for current session
export PATH="$HOME/miniconda3/bin:$PATH"

# Verify conda installation
if ! command -v conda &> /dev/null; then
    print_error "Conda installation failed. Please check the installation."
    exit 1
fi

print_info "Conda version: $(conda --version)"

# Create conda environment
ENV_NAME="resdsql"
print_info "Creating conda environment '$ENV_NAME' with Python 3.8.5..."

# Remove environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warn "Environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n "$ENV_NAME" -y
fi

conda create -n "$ENV_NAME" python=3.8.5 -y

# Activate conda environment
print_info "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install PyTorch with CUDA 11.3
print_info "Installing PyTorch 1.11.0 with CUDA 11.3..."
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# Verify PyTorch and CUDA
print_info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Get the project directory (assuming script is run from project root)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Install Python packages from requirements.txt
print_info "Installing Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    print_error "requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

# Install spacy and spacy model
print_info "Installing spacy and spacy model..."
pip install spacy==2.2.3
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

# Download NLTK data
print_info "Downloading NLTK punkt data..."
if [ -f "nltk_downloader.py" ]; then
    python nltk_downloader.py
else
    print_warn "nltk_downloader.py not found. Downloading punkt manually..."
    python -c "import nltk; nltk.download('punkt')"
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p eval_results
mkdir -p models
mkdir -p tensorboard_log
mkdir -p third_party
mkdir -p predictions

# Clone evaluation scripts
print_info "Cloning evaluation scripts..."
cd third_party

if [ ! -d "spider" ]; then
    git clone https://github.com/ElementAI/spider.git
else
    print_info "spider repository already exists. Skipping clone."
fi

if [ ! -d "test_suite" ]; then
    git clone https://github.com/ElementAI/test-suite-sql-eval.git
    mv ./test-suite-sql-eval ./test_suite
else
    print_info "test_suite repository already exists. Skipping clone."
fi

cd "$PROJECT_DIR"

# Install NatSQL requirements if needed
if [ -f "NatSQL/requirements.txt" ]; then
    print_info "Installing NatSQL requirements..."
    pip install -r NatSQL/requirements.txt
fi

# Optional: Download data, databases, and models from Google Drive
print_info ""
print_info "=========================================="
print_info "Optional: Download Data, Databases, and Models"
print_info "=========================================="
print_warn "This step downloads large files from Google Drive (several GB)."
print_warn "You can skip this and download manually later if preferred."
echo ""
read -p "Do you want to download data, databases, and models now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing gdown for Google Drive downloads..."
    pip install gdown
    
    # Download data
    if [ ! -d "data/spider" ] || [ ! -f "data.zip" ]; then
        print_info "Downloading data.zip from Google Drive (this may take a while)..."
        gdown --id 19tsgBGAxpagULSl9r85IFKIZb4kyBGGu -O data.zip || {
            print_error "Failed to download data.zip. You may need to download manually from:"
            print_info "https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing"
        }
        
        if [ -f "data.zip" ]; then
            print_info "Extracting data.zip..."
            unzip -q data.zip || print_error "Failed to extract data.zip"
        fi
    else
        print_info "Data already exists. Skipping download."
    fi
    
    # Download databases
    if [ ! -d "database" ] || [ ! -f "database.zip" ]; then
        print_info "Downloading database.zip from Google Drive (this may take a while)..."
        gdown --id 1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo -O database.zip || {
            print_error "Failed to download database.zip. You may need to download manually from:"
            print_info "https://drive.google.com/file/d/1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo/view?usp=share_link"
        }
        
        if [ -f "database.zip" ]; then
            print_info "Extracting database.zip..."
            unzip -q database.zip || print_error "Failed to extract database.zip"
        fi
    else
        print_info "Databases already exist. Skipping download."
    fi
    
    # Download model checkpoints (optional - user can choose which ones)
    print_info ""
    print_info "Model checkpoints are large files. You can download them now or later."
    print_info "For basic inference, you need:"
    print_info "  1. Cross-encoder: text2natsql_schema_item_classifier (or text2sql_schema_item_classifier)"
    print_info "  2. T5 model: text2natsql-t5-base (or text2sql-t5-base, or larger models)"
    echo ""
    read -p "Do you want to download model checkpoints now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Which cross-encoder do you want to download?"
        echo "  1) text2natsql_schema_item_classifier (recommended for NatSQL)"
        echo "  2) text2sql_schema_item_classifier (for SQL version)"
        echo "  3) Skip cross-encoder"
        read -p "Enter choice (1/2/3): " -n 1 -r
        echo ""
        
        if [[ $REPLY == "1" ]]; then
            print_info "Downloading text2natsql_schema_item_classifier..."
            mkdir -p models/text2natsql_schema_item_classifier
            gdown --id 1UWNj1ZADfKa1G5I4gBYCJeEQO6piMg4G -O models/text2natsql_schema_item_classifier.zip || {
                print_error "Failed to download. Download manually from:"
                print_info "https://drive.google.com/file/d/1UWNj1ZADfKa1G5I4gBYCJeEQO6piMg4G/view?usp=share_link"
            }
            if [ -f "models/text2natsql_schema_item_classifier.zip" ]; then
                cd models/text2natsql_schema_item_classifier
                unzip -q ../text2natsql_schema_item_classifier.zip
                rm ../text2natsql_schema_item_classifier.zip
                cd "$PROJECT_DIR"
            fi
        elif [[ $REPLY == "2" ]]; then
            print_info "Downloading text2sql_schema_item_classifier..."
            mkdir -p models/text2sql_schema_item_classifier
            gdown --id 1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo -O models/text2sql_schema_item_classifier.zip || {
                print_error "Failed to download. Download manually from:"
                print_info "https://drive.google.com/file/d/1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo/view?usp=share_link"
            }
            if [ -f "models/text2sql_schema_item_classifier.zip" ]; then
                cd models/text2sql_schema_item_classifier
                unzip -q ../text2sql_schema_item_classifier.zip
                rm ../text2sql_schema_item_classifier.zip
                cd "$PROJECT_DIR"
            fi
        fi
        
        print_info "Which T5 model do you want to download?"
        echo "  1) text2natsql-t5-base (recommended for 16GB RAM)"
        echo "  2) text2natsql-t5-large"
        echo "  3) text2natsql-t5-3b (requires more RAM)"
        echo "  4) text2sql-t5-base"
        echo "  5) text2sql-t5-large"
        echo "  6) text2sql-t5-3b"
        echo "  7) Skip T5 model"
        read -p "Enter choice (1-7): " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                print_info "Downloading text2natsql-t5-base..."
                mkdir -p models/text2natsql-t5-base
                gdown --id 1QyfSfHHrxfIM5X9gKUYNr_0ZRVvb1suV -O models/text2natsql-t5-base.zip || {
                    print_error "Failed to download. Download manually from:"
                    print_info "https://drive.google.com/file/d/1QyfSfHHrxfIM5X9gKUYNr_0ZRVvb1suV/view?usp=share_link"
                }
                if [ -f "models/text2natsql-t5-base.zip" ]; then
                    cd models/text2natsql-t5-base
                    unzip -q ../text2natsql-t5-base.zip
                    rm ../text2natsql-t5-base.zip
                    cd "$PROJECT_DIR"
                fi
                ;;
            2)
                print_info "Downloading text2natsql-t5-large..."
                mkdir -p models/text2natsql-t5-large
                gdown --id 1ZwFsH24_qKC3xwYdedPi6T_8argguWHe -O models/text2natsql-t5-large.zip || {
                    print_error "Failed to download. Download manually from:"
                    print_info "https://drive.google.com/file/d/1ZwFsH24_qKC3xwYdedPi6T_8argguWHe/view?usp=sharing"
                }
                if [ -f "models/text2natsql-t5-large.zip" ]; then
                    cd models/text2natsql-t5-large
                    unzip -q ../text2natsql-t5-large.zip
                    rm ../text2natsql-t5-large.zip
                    cd "$PROJECT_DIR"
                fi
                ;;
            3)
                print_warn "T5-3B is very large (~11GB) and requires significant RAM."
                print_info "Downloading text2natsql-t5-3b from OneDrive..."
                print_warn "OneDrive links require manual download. Please download from:"
                print_info "https://1drv.ms/u/s!Ak05bBUBFYiktcdziiE79xaeKtO6qg?e=e9424n"
                ;;
            4)
                print_info "Downloading text2sql-t5-base..."
                mkdir -p models/text2sql-t5-base
                gdown --id 1lqZ81f_fSZtg6BRcRw1-Ol-RJCcKRsmH -O models/text2sql-t5-base.zip || {
                    print_error "Failed to download. Download manually from:"
                    print_info "https://drive.google.com/file/d/1lqZ81f_fSZtg6BRcRw1-Ol-RJCcKRsmH/view?usp=sharing"
                }
                if [ -f "models/text2sql-t5-base.zip" ]; then
                    cd models/text2sql-t5-base
                    unzip -q ../text2sql-t5-base.zip
                    rm ../text2sql-t5-base.zip
                    cd "$PROJECT_DIR"
                fi
                ;;
            5)
                print_info "Downloading text2sql-t5-large..."
                mkdir -p models/text2sql-t5-large
                gdown --id 1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e -O models/text2sql-t5-large.zip || {
                    print_error "Failed to download. Download manually from:"
                    print_info "https://drive.google.com/file/d/1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e/view?usp=sharing"
                }
                if [ -f "models/text2sql-t5-large.zip" ]; then
                    cd models/text2sql-t5-large
                    unzip -q ../text2sql-t5-large.zip
                    rm ../text2sql-t5-large.zip
                    cd "$PROJECT_DIR"
                fi
                ;;
            6)
                print_info "Downloading text2sql-t5-3b..."
                mkdir -p models/text2sql-t5-3b
                gdown --id 1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11 -O models/text2sql-t5-3b.zip || {
                    print_error "Failed to download. Download manually from:"
                    print_info "https://drive.google.com/file/d/1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11/view?usp=sharing"
                }
                if [ -f "models/text2sql-t5-3b.zip" ]; then
                    cd models/text2sql-t5-3b
                    unzip -q ../text2sql-t5-3b.zip
                    rm ../text2sql-t5-3b.zip
                    cd "$PROJECT_DIR"
                fi
                ;;
        esac
    fi
    
    print_info "Download step completed."
else
    print_info "Skipping download step. You can download manually later:"
    print_info "  - Data: https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing"
    print_info "  - Databases: https://drive.google.com/file/d/1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo/view?usp=share_link"
    print_info "  - Models: See README.md for checkpoint download links"
fi

# Final verification
print_info "Running final verification..."
python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('✗ PyTorch not found')

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError:
    print('✗ Transformers not found')

try:
    import nltk
    print('✓ NLTK installed')
except ImportError:
    print('✗ NLTK not found')

try:
    import spacy
    print(f'✓ SpaCy {spacy.__version__}')
except ImportError:
    print('✗ SpaCy not found')
"

echo ""
echo "=========================================="
print_info "Setup completed successfully!"
echo "=========================================="
echo ""
print_info "Next steps:"
echo ""
echo "  1. Activate the conda environment:"
echo "     source ~/.bashrc"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. Verify GPU setup (if using GPU):"
echo "     nvidia-smi                    # Check if drivers are working"
echo "     python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
echo ""
echo "  3. If GPU is not available:"
echo "     - If using standard Ubuntu AMI, install drivers:"
echo "       sudo ubuntu-drivers autoinstall"
echo "       sudo reboot"
echo "     - Or consider using AWS Deep Learning AMI (recommended)"
echo ""
print_info "  4. Download data and databases:"
echo "     See README.md for download links"
echo "     Extract to ./data/ and ./database/ directories"
echo ""
print_info "  5. Download model checkpoints:"
echo "     See README.md for checkpoint download links"
echo "     Place checkpoints in the models/ directory"
echo ""
print_info "  6. Run inference:"
echo "     conda activate $ENV_NAME"
echo "     sh scripts/inference/infer_text2natsql.sh base spider"
echo ""
print_warn "IMPORTANT:"
echo "  - If NVIDIA drivers were installed, a reboot is required: sudo reboot"
echo "  - For best results, use AWS Deep Learning AMI which includes drivers and CUDA"
echo "  - The script will work without GPU (CPU mode), but will be slower"
echo ""
