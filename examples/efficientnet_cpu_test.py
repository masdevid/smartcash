#!/usr/bin/env python3
"""
Force CPU training for EfficientNet-B4 to avoid MPS memory limitations
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force CPU mode by monkey-patching device detection
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
torch.cuda.is_available = lambda: False

print("🖥️ Forcing CPU training mode")
print("🚫 MPS disabled to avoid memory constraints")

from smartcash.model.api.core import run_full_training_pipeline

def create_simple_progress_callback():
    """Simple progress callback for CPU training."""
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        percentage = (current / total) * 100 if total > 0 else 0
        phase_display = phase.replace('_', ' ').title()
        
        if phase in ['training_phase_1', 'training_phase_2']:
            if 'epoch' in kwargs:
                epoch = kwargs['epoch']
                batch_idx = kwargs.get('batch_idx', 0)
                batch_total = kwargs.get('batch_total', 1)
                metrics = kwargs.get('metrics', {})
                
                phase_num = "1" if phase == 'training_phase_1' else "2"
                
                if current == total and metrics:
                    loss = metrics.get('train_loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    print(f"📊 Phase {phase_num} - Epoch {epoch} COMPLETED:")
                    print(f"    Loss: Train={loss:.4f} Val={val_loss:.4f}")
                    
                    # Layer-specific metrics
                    for layer in ['layer_1', 'layer_2', 'layer_3']:
                        acc = metrics.get(f'{layer}_accuracy', 0)
                        if acc > 0:
                            prec = metrics.get(f'{layer}_precision', 0)
                            rec = metrics.get(f'{layer}_recall', 0)
                            f1 = metrics.get(f'{layer}_f1', 0)
                            print(f"    {layer.upper()}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
            else:
                print(f"🔄 {phase_display} ({percentage:.0f}%): {message}")
        else:
            if percentage == 100:
                print(f"✅ {phase_display}: {message}")
            else:
                print(f"🔄 {phase_display} ({percentage:.0f}%): {message}")
    
    return progress_callback

def main():
    """Run EfficientNet-B4 training on CPU."""
    print("=" * 80)
    print("🚀 SmartCash EfficientNet-B4 CPU Training")
    print("=" * 80)
    print("📋 Configuration:")
    print("   • Backbone: efficientnet_b4")
    print("   • Phase 1 epochs: 1 (frozen backbone)")
    print("   • Phase 2 epochs: 1 (fine-tuning)")
    print("   • Device: CPU (forced)")
    print("   • Batch size: 8")
    print("=" * 80)
    
    try:
        # Create progress callback
        progress_callback = create_simple_progress_callback()
        
        print("🚀 Starting CPU training pipeline...")
        print("⚠️  Note: CPU training will be slower than GPU training")
        print()
        
        result = run_full_training_pipeline(
            backbone='efficientnet_b4',
            phase_1_epochs=1,
            phase_2_epochs=1,
            checkpoint_dir='data/checkpoints',
            progress_callback=progress_callback,
            verbose=True,
            force_cpu=True,  # Force CPU mode
            training_mode='two_phase',  # Use traditional two-phase training
            # Training configuration
            loss_type='uncertainty_multi_task',
            head_lr_p1=0.001,
            head_lr_p2=0.0001,
            backbone_lr=1e-5,
            batch_size=8,
            # Early stopping settings
            early_stopping_enabled=True,
            early_stopping_patience=15,
            early_stopping_metric='val_map50',
            early_stopping_mode='max',
            early_stopping_min_delta=0.001
        )
        
        # Process results
        if result.get('success'):
            print("\n" + "=" * 80)
            print("🎉 EfficientNet-B4 CPU Training Completed Successfully!")
            print("=" * 80)
            
            # Display results
            pipeline_summary = result.get('pipeline_summary', {})
            total_duration = pipeline_summary.get('total_duration', 0)
            
            print(f"📊 Training Summary:")
            print(f"   • Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"   • Device used: CPU")
            print(f"   • Phases completed: {pipeline_summary.get('phases_completed', 0)}/6")
            
            # Training results
            training_result = result.get('final_training_result', {})
            if training_result.get('success'):
                best_metrics = training_result.get('best_metrics', {})
                
                print(f"\n🏆 Training Results:")
                print(f"   • Final train loss: {best_metrics.get('train_loss', 0):.4f}")
                print(f"   • Final val loss: {best_metrics.get('val_loss', 0):.4f}")
                
                # Layer metrics
                print(f"\n📊 Layer Performance:")
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    acc = best_metrics.get(f'{layer}_accuracy', 0)
                    if acc > 0:
                        f1 = best_metrics.get(f'{layer}_f1', 0)
                        print(f"   • {layer.upper()}: Acc={acc:.4f} F1={f1:.4f}")
            
            return 0
            
        else:
            print("\n" + "=" * 80)
            print("❌ EfficientNet-B4 CPU Training Failed")
            print("=" * 80)
            error_msg = result.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())