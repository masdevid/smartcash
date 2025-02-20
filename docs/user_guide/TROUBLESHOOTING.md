# 🔧 Troubleshooting Guide

## 📋 Common Issues

### 1. Installation Issues

#### CUDA Errors
```
Problem: CUDA not found
Solution:
1. Check CUDA installation
   nvidia-smi
2. Verify PyTorch CUDA
   python -c "import torch; print(torch.cuda.is_available())"
3. Reinstall PyTorch with CUDA
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

#### Dependencies
```
Problem: Package conflicts
Solution:
1. Create fresh environment
   conda create -n smartcash python=3.9
2. Install requirements in order
   pip install -r requirements.txt
```

### 2. Training Issues

#### Out of Memory
```
Problem: CUDA out of memory
Solution:
1. Reduce batch size
   --batch-size 8
2. Enable AMP
   --amp
3. Use gradient checkpointing
   --gradient-checkpointing
```

#### Loss Not Converging
```
Problem: Training loss stuck
Solution:
1. Check learning rate
   --lr 0.001
2. Verify data
   python tools/validate_dataset.py
3. Monitor gradients
   --wandb
```

### 3. Inference Issues

#### Slow Inference
```
Problem: Low FPS
Solution:
1. Use TensorRT
   python export.py --include tensorrt
2. Enable FP16
   --half
3. Optimize batch size
   --batch-size 4
```

#### Poor Detection
```
Problem: Low accuracy
Solution:
1. Adjust confidence threshold
   --conf-thres 0.25
2. Tune NMS threshold
   --iou-thres 0.45
3. Check input resolution
   --img-size 640
```

## 🔍 Debugging Tools

### 1. Model Debugging
```python
# Enable debug mode
python train.py --debug

# Profile model
python -m torch.utils.bottleneck train.py
```

### 2. Data Debugging
```python
# Validate dataset
python tools/check_dataset.py

# Visualize augmentations
python tools/vis_augmentations.py
```

## 📊 System Checks

### 1. GPU Status
```bash
# Check GPU usage
nvidia-smi -l 1

# Monitor processes
ps aux | grep python
```

### 2. Disk Space
```bash
# Check space
df -h

# Find large files
find . -type f -size +100M
```

## 🔄 Common Workflows

### 1. Reset Environment
```bash
# Clear cache
rm -rf ~/.cache/torch
rm -rf runs/*

# Reset environment
conda deactivate
conda env remove -n smartcash
conda create -n smartcash python=3.9
```

### 2. Update System
```bash
# Update CUDA
sudo apt update
sudo apt install cuda-toolkit-11-7

# Update PyTorch
pip install --upgrade torch torchvision
```

## 🛠️ Advanced Debugging

### 1. Memory Profiling
```python
# Profile memory
from memory_profiler import profile

@profile
def train():
    ...
```

### 2. CUDA Profiling
```bash
# Use nvprof
nvprof python train.py

# Use PyTorch profiler
python -m torch.utils.bottleneck train.py
```

## 📝 Logging

### 1. Enable Detailed Logging
```python
# Set logging level
import logging
logging.basicConfig(level=logging.DEBUG)

# Log to file
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG
)
```

### 2. System Logs
```bash
# Check system logs
journalctl -u nvidia-persistenced

# Check CUDA logs
cat /var/log/cuda-installer.log
```

## 🔍 Performance Issues

### 1. CPU Performance
```bash
# Check CPU usage
top -u $USER

# Monitor temperature
sensors
```

### 2. GPU Performance
```bash
# Check GPU stats
nvidia-smi dmon

# Monitor power usage
nvidia-smi -q -d POWER
```

## 🌐 Network Issues

### 1. Download Issues
```bash
# Test connection
ping google.com

# Check DNS
nslookup google.com
```

### 2. API Issues
```bash
# Test API
curl -X GET http://localhost:8000/health

# Check ports
netstat -tulpn
```

## 📁 File Issues

### 1. Permission Issues
```bash
# Fix permissions
chmod -R 755 .
chown -R $USER:$USER .

# Check permissions
ls -la
```

### 2. Path Issues
```bash
# Check paths
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

# Add to path
export PYTHONPATH=$PYTHONPATH:/path/to/smartcash
```

## 🆘 Getting Help

1. Check documentation
2. Search issues on GitHub
3. Join Discord community
4. Open new issue

## 🔄 Recovery Steps

1. Backup important files
2. Clean installation
3. Restore from backup
4. Verify functionality

## 📈 Next Steps

1. [Return to training](TRAINING.md)
2. [Check evaluation](EVALUATION.md)
3. [Review architecture](../technical/ARSITEKTUR.md)
