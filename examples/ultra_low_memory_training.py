#!/usr/bin/env python3
"""
Ultra-low memory training script for systems with severe memory constraints
Specifically designed for MPS out-of-memory issues on Mac systems
Uses extreme memory conservation techniques
"""

import sys
import os
import torch
import gc
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import run_full_training_pipeline

def setup_ultra_low_memory():
    """Setup ultra-conservative memory settings"""
    # Force CPU if MPS memory is insufficient
    if torch.backends.mps.is_available():
        print("🚨 Ultra-low memory mode: Forcing CPU to avoid MPS OOM")
        # Monkey patch to disable MPS
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    
    # Ultra-conservative CPU settings
    torch.set_num_threads(2)  # Minimal threading
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # Disable CUDA
    torch.cuda.is_available = lambda: False
    
    print("🔧 Ultra-low memory settings applied:")
    print("   • Device: CPU (forced)")
    print("   • Threads: 2 (minimal)")
    print("   • MPS: Disabled")
    print("   • CUDA: Disabled")

def aggressive_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    for _ in range(10):
        gc.collect()
    print("🧹 Ultra-aggressive memory cleanup completed")

def create_ultra_minimal_progress_callback():
    """Minimal progress callback to reduce memory overhead"""
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        if phase in ['training_phase_1', 'training_phase_2']:
            if 'epoch' in kwargs:
                epoch = kwargs['epoch']
                phase_num = "1" if phase == 'training_phase_1' else "2"
                
                if current == total:
                    metrics = kwargs.get('metrics', {})
                    train_loss = metrics.get('train_loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    
                    print(f"\n✅ Phase {phase_num} - Epoch {epoch} Complete:")
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss:   {val_loss:.4f}")
                    
                    # Ultra-aggressive cleanup after each epoch
                    aggressive_memory_cleanup()
                else:
                    percentage = (current / total) * 100 if total > 0 else 0
                    print(f"⏳ Phase {phase_num} - Epoch {epoch}: {percentage:.0f}%", end='\r')
        else:
            if current >= total:
                print(f"✅ {phase.replace('_', ' ').title()}: Complete")
    
    return progress_callback

def main():
    """Run ultra-low memory training"""
    parser = argparse.ArgumentParser(description="Ultra-low memory training for severe memory constraints")
    parser.add_argument('--backbone', choices=['efficientnet_b4', 'cspdarknet'], default='efficientnet_b4')
    parser.add_argument('--epochs', type=int, default=1, help='Total epochs (ultra-conservative)')
    args = parser.parse_args()
    
    print("="*80)
    print("🚨 ULTRA-LOW MEMORY TRAINING MODE")
    print("="*80)
    print("⚠️  WARNING: This mode sacrifices performance for memory conservation")
    print(f"🎯 Backbone: {args.backbone}")
    print(f"🔄 Epochs: {args.epochs} (minimal)")
    print("="*80)
    
    # Setup ultra-low memory mode
    setup_ultra_low_memory()
    
    # Pre-training cleanup
    aggressive_memory_cleanup()
    
    try:
        progress_callback = create_ultra_minimal_progress_callback()
        
        print("🚀 Starting ultra-low memory training...")
        print("⚠️  This will be slower but should avoid memory errors")
        
        result = run_full_training_pipeline(
            backbone=args.backbone,
            phase_1_epochs=args.epochs,
            phase_2_epochs=0,  # Skip phase 2 to minimize memory usage
            checkpoint_dir='data/checkpoints',
            progress_callback=progress_callback,
            verbose=False,  # Reduce logging overhead
            force_cpu=True,
            training_mode='single_phase',  # Simplified training mode
            
            # Ultra-conservative memory settings
            batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=16,  # High accumulation for effective batch size
            use_mixed_precision=False,
            
            # Minimal training configuration
            loss_type='uncertainty_multi_task',
            head_lr_p1=0.001,
            backbone_lr=1e-5,
            
            # Ultra-conservative early stopping
            early_stopping_enabled=True,
            early_stopping_patience=5,
            early_stopping_metric='val_map50',
            early_stopping_mode='max',
            early_stopping_min_delta=0.005,
            
            # Memory-saving data loading
            dataloader_num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            
            # Gradient management
            max_grad_norm=0.5,
            weight_decay=0.001
        )
        
        # Final cleanup
        aggressive_memory_cleanup()
        
        if result.get('success'):
            print("\n" + "="*80)
            print("🎉 Ultra-Low Memory Training Completed!")
            print("="*80)
            
            training_result = result.get('final_training_result', {})
            if training_result.get('success'):
                best_metrics = training_result.get('best_metrics', {})
                print(f"📊 Results:")
                print(f"   • Train Loss: {best_metrics.get('train_loss', 0):.4f}")
                print(f"   • Val Loss:   {best_metrics.get('val_loss', 0):.4f}")
                
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    acc = best_metrics.get(f'{layer}_accuracy', 0)
                    if acc > 0:
                        print(f"   • {layer.upper()}: {acc:.3f}")
            
            print("\n💡 Memory Conservation Results:")
            print("   • ✅ Avoided MPS out-of-memory errors")
            print("   • ✅ Ultra-conservative CPU training completed")
            print("   • ⚠️  Performance traded for memory stability")
            
            return 0
        else:
            print("\n❌ Ultra-Low Memory Training Failed")
            error_msg = result.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        print("🚨 Even ultra-low memory mode failed!")
        print("💡 Your system may need more RAM or try a smaller model")
        aggressive_memory_cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main())