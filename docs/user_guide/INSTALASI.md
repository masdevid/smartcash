# 🛠️ Instalasi & Setup

## 📋 Requirements

- Python 3.9+
- CUDA 11.7+ (untuk GPU)
- 8GB RAM (minimum)
- 20GB disk space

## 🔧 Setup Environment

### 1. Create Conda Environment

```bash
# Create environment
conda create -n smartcash python=3.9
conda activate smartcash

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 2. Install Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/smartcash.git
cd smartcash

# Install requirements
pip install -r requirements.txt
```

### 3. Setup Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env
```

Required variables:
```env
ROBOFLOW_API_KEY=your_api_key
MODEL_PATH=/path/to/model
DATASET_PATH=/path/to/dataset
```

## 🔍 Verify Installation

```bash
# Test GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model inference
python scripts/test_inference.py
```

## 📁 Directory Structure

```
smartcash/
├── configs/           # Configuration files
├── data/             # Dataset directory
├── models/           # Trained models
├── scripts/          # Utility scripts
└── logs/             # Log files
```

## 🔄 Update & Maintenance

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Clear Cache

```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +

# Clear model cache
rm -rf ~/.cache/torch/hub/checkpoints/*
```

## 🐛 Common Issues

### CUDA Issues

1. Check CUDA version:
```bash
nvidia-smi
```

2. Verify PyTorch CUDA:
```python
import torch
print(torch.version.cuda)
```

3. If mismatch, reinstall PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Memory Issues

1. Reduce batch size in `configs/base_config.yaml`
2. Enable gradient checkpointing
3. Use mixed precision training

### Permission Issues

```bash
# Fix permissions
chmod +x scripts/*.sh
chmod -R 755 data/
```

## 🔄 Version Control

### Git Setup

```bash
# Configure Git
git config user.name "Your Name"
git config user.email "your@email.com"

# Setup LFS
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

## 📚 Next Steps

1. [Prepare dataset](../dataset/README.md)
2. [Train model](TRAINING.md)
3. [Evaluate results](EVALUATION.md)

## 🆘 Support

- [Open issue](https://github.com/yourusername/smartcash/issues)
- [Documentation](../README.md)
- [Troubleshooting](TROUBLESHOOTING.md)
