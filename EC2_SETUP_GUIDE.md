# EC2 Ubuntu Setup Guide for RESDSQL

This guide will help you set up RESDSQL on an EC2 Ubuntu instance (g4dn.xlarge with GPU).

## Prerequisites

- EC2 instance: g4dn.xlarge (or similar GPU instance)
- Ubuntu 20.04 or 22.04
- SSH access to the instance

## Recommended: Use AWS Deep Learning AMI

**We STRONGLY recommend using AWS Deep Learning AMI (DLAMI)** which already includes:
- NVIDIA drivers
- CUDA toolkit
- cuDNN
- Other ML frameworks

This will save significant setup time and avoid compatibility issues.

### Launching with Deep Learning AMI:

1. Go to EC2 Console → Launch Instance
2. Search for "Deep Learning AMI" in the AMI search
3. Select "Deep Learning AMI GPU PyTorch" (or similar)
4. Choose g4dn.xlarge instance type
5. Configure security group to allow SSH (port 22)
6. Launch and connect via SSH

## Setup Steps

### Step 1: Connect to Your EC2 Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 2: Clone the Repository

```bash
# Install git if not already installed
sudo apt-get update
sudo apt-get install -y git

# Clone the repository
git clone <your-repo-url>
cd RESDSQL
```

Or upload the project files using SCP:

```bash
# From your local machine
scp -i your-key.pem -r /path/to/RESDSQL ubuntu@your-ec2-ip:~/
```

### Step 3: Run the Setup Script

```bash
# Make the script executable
chmod +x setup_ec2_ubuntu.sh

# Run the setup script
./setup_ec2_ubuntu.sh
```

The script will:
- Install system dependencies
- Install Miniconda
- Create conda environment with Python 3.8.5
- Install PyTorch 1.11.0 with CUDA 11.3 support
- Install all Python packages
- Set up project directories
- Clone evaluation scripts

**Note:** The script takes approximately 15-30 minutes to complete.

### Step 4: Verify GPU Setup

After the script completes:

```bash
# Activate conda environment
source ~/.bashrc
conda activate resdsql

# Check NVIDIA drivers
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If `torch.cuda.is_available()` returns `False`:
- If using standard Ubuntu AMI: Install drivers manually:
  ```bash
  sudo ubuntu-drivers autoinstall
  sudo reboot
  ```
- Or use Deep Learning AMI (recommended)

### Step 5: Download Data, Databases, and Models

**Option A: Automatic Download (Recommended)**

The setup script can automatically download data, databases, and models from Google Drive. When the script asks:

```bash
Do you want to download data, databases, and models now? (y/n)
```

Answer `y` and follow the prompts to select which models to download.

**Option B: Manual Download**

If you prefer to download manually:

1. **Spider Data**: [Download Link](https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing)
2. **Databases**: [Download Link](https://drive.google.com/file/d/1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo/view?usp=share_link)

Upload to EC2 instance:

```bash
# From your local machine
scp -i your-key.pem data.zip ubuntu@your-ec2-ip:~/RESDSQL/
scp -i your-key.pem database.zip ubuntu@your-ec2-ip:~/RESDSQL/
```

Extract on EC2:

```bash
cd ~/RESDSQL
unzip data.zip
unzip database.zip
```

### Step 6: Download Model Checkpoints (if not done automatically)

Download the required model checkpoints (see README.md for links):

**For NatSQL version (recommended):**
- Cross-encoder: `text2natsql_schema_item_classifier`
- T5 model: `text2natsql-t5-base` (or large/3b)

**For SQL version:**
- Cross-encoder: `text2sql_schema_item_classifier`
- T5 model: `text2sql-t5-base` (or large/3b)

Upload checkpoints to EC2:

```bash
# From your local machine
scp -i your-key.pem -r models/ ubuntu@your-ec2-ip:~/RESDSQL/
```

Or download directly on EC2 using `gdown` or `wget` (if links are direct).

### Step 7: Run Inference

```bash
# Activate environment
conda activate resdsql

# Navigate to project directory
cd ~/RESDSQL

# Run inference (example: T5-Base with NatSQL on Spider)
sh scripts/inference/infer_text2natsql.sh base spider
```

## Troubleshooting

### GPU Not Available

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   If this fails, install drivers:
   ```bash
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

2. **Check CUDA:**
   ```bash
   nvcc --version
   ```
   If not found, use Deep Learning AMI or install CUDA manually.

3. **Verify PyTorch CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Out of Memory Errors

For g4dn.xlarge (16GB RAM):
- Use T5-Base model (recommended)
- Reduce batch size in inference scripts
- T5-Large may work but is tight
- T5-3B is too large for 16GB RAM

### Slow Performance

- Ensure GPU is being used (check `nvidia-smi` during inference)
- Use appropriate batch size for your GPU memory
- Consider using larger instance (g4dn.2xlarge or g4dn.4xlarge) for larger models

### Conda Environment Issues

If conda commands don't work:

```bash
source ~/.bashrc
# Or
export PATH="$HOME/miniconda3/bin:$PATH"
```

## Instance Recommendations

| Model | Instance Type | RAM | GPU Memory | Recommendation |
|-------|--------------|-----|------------|----------------|
| T5-Base | g4dn.xlarge | 16GB | 16GB | ✅ Recommended |
| T5-Large | g4dn.2xlarge | 32GB | 16GB | ✅ Good |
| T5-3B | g4dn.4xlarge | 64GB | 16GB | ✅ Best |

## Quick Reference

```bash
# Activate environment
conda activate resdsql

# Check GPU
nvidia-smi

# Run inference
sh scripts/inference/infer_text2natsql.sh base spider

# Check results
cat predictions/Spider-dev/resdsql_base_natsql/pred.sql
```

## Additional Resources

- Main README: `README.md`
- Inference Guide: `INFERENCE_GUIDE.md`
- Setup Notes: `SETUP_NOTES.md`
