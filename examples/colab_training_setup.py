#!/usr/bin/env python3
"""
Google Colab Training Setup Script
Generates a complete Colab notebook for cloud training with GPU
Solves local memory issues by using cloud resources
"""

def generate_colab_notebook():
    """Generate a complete Google Colab notebook for training"""
    
    notebook_content = """
# SmartCash EfficientNet-B4 Training on Google Colab
# Solves local Mac memory issues with cloud GPU training

## 🚀 Setup Environment
```python
# Install required packages
!pip install timm torchvision tqdm psutil

# Clone your repository (replace with your actual repo)
!git clone https://github.com/yourusername/smartcash.git
%cd smartcash

# Install project dependencies
!pip install -r requirements.txt  # if you have one
```

## 🔧 Import and Setup
```python
import torch
import gc
import os
from pathlib import Path

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Setup paths
import sys
sys.path.append('/content/smartcash')

from smartcash.model.api.core import run_full_training_pipeline
```

## 📁 Upload Your Data
```python
# Upload your dataset to Colab
# Option 1: Upload directly
from google.colab import files
uploaded = files.upload()

# Option 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Option 3: Download from cloud storage
# !wget your-dataset-url
# !unzip your-dataset.zip
```

## 🎯 Training Configuration
```python
def create_colab_progress_callback():
    \"\"\"Progress callback optimized for Colab\"\"\"
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        percentage = (current / total) * 100 if total > 0 else 0
        
        if phase in ['training_phase_1', 'training_phase_2']:
            if 'epoch' in kwargs:
                epoch = kwargs['epoch']
                phase_num = "1" if phase == 'training_phase_1' else "2"
                
                if current == total and kwargs.get('metrics'):
                    metrics = kwargs['metrics']
                    print(f"📊 Phase {phase_num} - Epoch {epoch} Complete:")
                    print(f"   Train Loss: {metrics.get('train_loss', 0):.4f}")
                    print(f"   Val Loss: {metrics.get('val_loss', 0):.4f}")
                    
                    # GPU memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    # Layer metrics
                    for layer in ['layer_1', 'layer_2', 'layer_3']:
                        acc = metrics.get(f'{layer}_accuracy', 0)
                        if acc > 0:
                            print(f"   {layer}: Acc={acc:.3f}")
                else:
                    print(f"🔄 Phase {phase_num} - Epoch {epoch}: {percentage:.0f}%")
        else:
            if current >= total:
                print(f"✅ {phase.replace('_', ' ').title()}: Complete")
    
    return progress_callback

# Training configuration optimized for Colab
training_config = {
    'backbone': 'efficientnet_b4',
    'phase_1_epochs': 5,  # More epochs since we have GPU
    'phase_2_epochs': 5,
    'checkpoint_dir': '/content/checkpoints',
    'progress_callback': create_colab_progress_callback(),
    'verbose': True,
    'force_cpu': False,  # Use GPU!
    'training_mode': 'two_phase',
    
    # GPU-optimized settings
    'batch_size': 8,  # Larger batch size with GPU
    'gradient_accumulation_steps': 2,  # Less accumulation needed
    'use_mixed_precision': True,  # GPU supports mixed precision
    
    # Training parameters
    'loss_type': 'uncertainty_multi_task',
    'head_lr_p1': 0.001,
    'head_lr_p2': 0.0001,
    'backbone_lr': 1e-5,
    
    # GPU-friendly early stopping
    'early_stopping_enabled': True,
    'early_stopping_patience': 10,
    'early_stopping_metric': 'val_map50',
    'early_stopping_mode': 'max',
    'early_stopping_min_delta': 0.001,
    
    # GPU data loading
    'dataloader_num_workers': 2,  # Can use workers on GPU
    'pin_memory': True,  # GPU benefits from pinned memory
    'persistent_workers': True,
    
    # Gradient management
    'max_grad_norm': 1.0,
    'weight_decay': 0.0005
}
```

## 🚀 Start Training
```python
# Create checkpoints directory
os.makedirs('/content/checkpoints', exist_ok=True)

print("🚀 Starting GPU training on Google Colab...")
print("="*60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*60)

# Run training
result = run_full_training_pipeline(**training_config)

# Process results
if result.get('success'):
    print("\\n🎉 COLAB TRAINING COMPLETED SUCCESSFULLY!")
    
    training_result = result.get('final_training_result', {})
    if training_result.get('success'):
        best_metrics = training_result.get('best_metrics', {})
        
        print("\\n📊 Final Results:")
        print(f"Train Loss: {best_metrics.get('train_loss', 0):.4f}")
        print(f"Val Loss: {best_metrics.get('val_loss', 0):.4f}")
        
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            acc = best_metrics.get(f'{layer}_accuracy', 0)
            if acc > 0:
                f1 = best_metrics.get(f'{layer}_f1', 0)
                print(f"{layer}: Acc={acc:.4f} F1={f1:.4f}")
    
    print("\\n📁 Download your trained model:")
    print("Checkpoints saved in: /content/checkpoints/")
    
else:
    print("❌ Training failed:", result.get('error', 'Unknown error'))
```

## 💾 Download Results
```python
# Zip and download checkpoints
!zip -r trained_model.zip /content/checkpoints/

from google.colab import files
files.download('trained_model.zip')

print("✅ Model downloaded! You can now use it locally.")
```

## 🔧 Advanced GPU Monitoring (Optional)
```python
# Monitor GPU usage during training
!pip install gpustat
!gpustat --watch
```
"""

    return notebook_content

def create_local_colab_helper():
    """Create a helper script to generate Colab setup"""
    
    helper_content = '''#!/usr/bin/env python3
"""
Colab Training Helper - Generates notebook content for cloud training
"""

def main():
    print("🌩️  GOOGLE COLAB TRAINING SETUP")
    print("="*60)
    print("Your Mac is running out of memory for local training.")
    print("Google Colab provides free GPU with 15GB+ memory!")
    print("="*60)
    
    print("\\n📝 STEP-BY-STEP GUIDE:")
    print("1. Go to: https://colab.research.google.com")
    print("2. Click: New notebook")
    print("3. Runtime > Change runtime type > GPU")
    print("4. Copy the code blocks below into cells")
    print("5. Run each cell in order")
    
    print("\\n🚀 ADVANTAGES OF COLAB TRAINING:")
    print("• Free GPU access (T4 with 15GB memory)")
    print("• No local memory limitations")
    print("• Faster training than CPU")
    print("• No process killing issues")
    print("• Can train larger models")
    
    print("\\n💡 CODE TO COPY INTO COLAB:")
    print("="*60)
    
    notebook = generate_colab_notebook()
    print(notebook)
    
    print("\\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("1. Copy the code above into Google Colab")
    print("2. Upload your dataset to Colab")  
    print("3. Run the training")
    print("4. Download the trained model")
    print("5. Use the model locally for inference")
    print("="*60)

if __name__ == "__main__":
    main()
'''
    
    return helper_content

def main():
    """Main function to create Colab setup"""
    print("🌩️  COLAB TRAINING SETUP GENERATOR")
    print("="*60)
    
    # Generate the helper script
    helper_content = create_local_colab_helper()
    
    # Write to file
    with open('colab_training_helper.py', 'w') as f:
        f.write(helper_content)
    
    print("✅ Generated: colab_training_helper.py")
    print("\\n🚀 To see the complete Colab setup:")
    print("python colab_training_helper.py")
    
    print("\\n💡 QUICK START:")
    print("1. Go to https://colab.research.google.com")
    print("2. Create new notebook with GPU runtime")
    print("3. Upload your SmartCash project")
    print("4. Copy training code from helper script")
    print("5. Train with 15GB+ GPU memory!")

if __name__ == "__main__":
    main()
'''

def main():
    """Create the complete Colab setup"""
    # Generate notebook content
    notebook = generate_colab_notebook()
    
    # Save to file
    with open('/Users/masdevid/Projects/smartcash/COLAB_TRAINING_NOTEBOOK.md', 'w') as f:
        f.write("# Google Colab Training Notebook\n")
        f.write("Copy the code blocks below into Google Colab cells\n\n")
        f.write(notebook)
    
    print("✅ Generated complete Colab training notebook!")
    print("📁 File: COLAB_TRAINING_NOTEBOOK.md")
    
    return notebook

if __name__ == "__main__":
    main()