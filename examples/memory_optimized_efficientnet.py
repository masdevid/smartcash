#!/usr/bin/env python3
"""
Simplified backbone training example with automatic memory optimization
Demonstrates training from scratch for EfficientNet-B4 and CSPDarkNet (YOLOv5s) backbones
Uses integrated memory optimization from core services
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import run_full_training_pipeline


def create_tqdm_progress_callback():
    """Progress callback with tqdm progress bars"""
    phase_bars = {}
    
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        # Simple device info for display
        import torch
        if torch.backends.mps.is_available():
            device_info = "MPS"
        elif torch.cuda.is_available():
            device_info = "CUDA"
        else:
            device_info = "CPU"
        
        # Create or update tqdm bar for this phase
        if phase not in phase_bars:
            phase_display = phase.replace('_', ' ').title()
            phase_bars[phase] = tqdm(
                total=total,
                desc=f"🔄 {phase_display} [{device_info}]",
                unit="step",
                ncols=100,
                position=len(phase_bars)
            )
        
        bar = phase_bars[phase]
        
        # Update progress
        if current > bar.n:
            bar.update(current - bar.n)
        
        # Update description with message
        if message:
            phase_display = phase.replace('_', ' ').title()
            bar.set_description(f"🔄 {phase_display} [{device_info}]: {message}")
        
        # Handle training phase specifics
        if phase in ['training_phase_1', 'training_phase_2']:
            if 'epoch' in kwargs:
                epoch = kwargs['epoch']
                metrics = kwargs.get('metrics', {})
                phase_num = "1" if phase == 'training_phase_1' else "2"
                
                # Update progress bar description with current epoch
                bar.set_description(f"🔄 Phase {phase_num} - Epoch {epoch}/{kwargs.get('total_epochs', '?')} [{device_info}]")
                
                if current == total and metrics:
                    # Close the progress bar and print detailed epoch results
                    bar.close()
                    
                    tqdm.write("\n" + "="*60)
                    tqdm.write(f"📊 PHASE {phase_num} - EPOCH {epoch} COMPLETED [{device_info}]")
                    tqdm.write("="*60)
                    
                    # Training and validation losses
                    train_loss = metrics.get('train_loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    tqdm.write(f"🎯 Loss Metrics:")
                    tqdm.write(f"   • Train Loss: {train_loss:.6f}")
                    tqdm.write(f"   • Val Loss:   {val_loss:.6f}")
                    tqdm.write(f"   • Loss Diff:  {abs(train_loss - val_loss):.6f}")
                    
                    # Overall mAP metrics
                    map50 = metrics.get('val_map50', 0)
                    map5095 = metrics.get('val_map50_95', 0)
                    if map50 > 0 or map5095 > 0:
                        tqdm.write(f"\n📈 Detection Metrics:")
                        tqdm.write(f"   • mAP@0.5:     {map50:.4f}")
                        tqdm.write(f"   • mAP@0.5:0.95: {map5095:.4f}")
                    
                    # Layer-specific detailed metrics
                    tqdm.write(f"\n🎯 Layer Performance Details:")
                    layer_found = False
                    for layer in ['layer_1', 'layer_2', 'layer_3']:
                        acc = metrics.get(f'{layer}_accuracy', 0)
                        if acc > 0:
                            layer_found = True
                            prec = metrics.get(f'{layer}_precision', 0)
                            rec = metrics.get(f'{layer}_recall', 0)
                            f1 = metrics.get(f'{layer}_f1', 0)
                            map50_layer = metrics.get(f'{layer}_map50', 0)
                            
                            tqdm.write(f"   • {layer.upper()}:")
                            tqdm.write(f"     - Accuracy:  {acc:.4f}")
                            tqdm.write(f"     - Precision: {prec:.4f}")
                            tqdm.write(f"     - Recall:    {rec:.4f}")
                            tqdm.write(f"     - F1-Score:  {f1:.4f}")
                            if map50_layer > 0:
                                tqdm.write(f"     - mAP@0.5:   {map50_layer:.4f}")
                    
                    if not layer_found:
                        tqdm.write("   • No layer-specific metrics available yet")
                    
                    # Learning rate and timing info
                    lr = metrics.get('learning_rate', 0)
                    epoch_time = metrics.get('epoch_time', 0)
                    if lr > 0 or epoch_time > 0:
                        tqdm.write(f"\n⚙️ Training Info:")
                        if lr > 0:
                            tqdm.write(f"   • Learning Rate: {lr:.2e}")
                        if epoch_time > 0:
                            tqdm.write(f"   • Epoch Time:    {epoch_time:.1f}s")
                    
                    # Best metrics tracking
                    best_val_loss = metrics.get('best_val_loss', 0)
                    best_map50 = metrics.get('best_val_map50', 0)
                    if best_val_loss > 0 or best_map50 > 0:
                        tqdm.write(f"\n🏆 Best So Far:")
                        if best_val_loss > 0:
                            tqdm.write(f"   • Best Val Loss: {best_val_loss:.6f}")
                        if best_map50 > 0:
                            tqdm.write(f"   • Best mAP@0.5:  {best_map50:.4f}")
                    
                    tqdm.write("="*60 + "\n")
                    
                    # Remove from tracking
                    del phase_bars[phase]
                else:
                    # Update progress bar with batch info if available
                    batch_idx = kwargs.get('batch_idx', 0)
                    total_batches = kwargs.get('total_batches', 0)
                    if batch_idx > 0 and total_batches > 0:
                        bar.set_description(f"🔄 Phase {phase_num} - Epoch {epoch} - Batch {batch_idx}/{total_batches} [{device_info}]")
        
        # Complete phase handling
        if current >= total:
            bar.close()
            phase_display = phase.replace('_', ' ').title()
            tqdm.write(f"✅ {phase_display} [{device_info}]: Completed")
            if phase in phase_bars:
                del phase_bars[phase]
    
    return progress_callback

def parse_arguments():
    """Parse command line arguments for backbone configuration"""
    parser = argparse.ArgumentParser(
        description="Simplified training example for EfficientNet-B4 and CSPDarkNet (YOLOv5s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Backbone selection
    parser.add_argument(
        '--backbone', '-b',
        type=str,
        default='efficientnet_b4',
        choices=['efficientnet_b4', 'cspdarknet'],
        help='Backbone model to use for training'
    )
    
    # Training configuration  
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for single phase training')
    parser.add_argument('--disable-tqdm', action='store_true', help='Disable tqdm progress bars')
    
    return parser.parse_args()


def main():
    """Run simplified backbone training with automatic memory optimization"""
    args = parse_arguments()
    
    print("=" * 80)
    print(f"🧠 SmartCash Training Example - {args.backbone.upper()}")
    print("=" * 80)
    print(f"   • Backbone: {args.backbone}")
    print(f"   • Epochs: {args.epochs} (single phase multi-layer training)")
    print(f"   • Pretrained: No (training from scratch)")
    print(f"   • Memory optimization: Automatic")
    print(f"   • Progress bars: {'Disabled' if args.disable_tqdm else 'Enabled'}")
    print("=" * 80)
    
    try:
        # Create progress callback
        progress_callback = None if args.disable_tqdm else create_tqdm_progress_callback()
        
        print("🚀 Starting training pipeline...")
        
        result = run_full_training_pipeline(
            backbone=args.backbone,
            phase_1_epochs=args.epochs,  # Single phase training
            phase_2_epochs=0,  # No second phase
            checkpoint_dir='data/checkpoints',
            progress_callback=progress_callback,
            verbose=True,
            training_mode='single_phase',  # Single phase multi-layer training
            single_phase_layer_mode='multi',  # Multi-layer detection
            single_phase_freeze_backbone=False,  # Don't freeze backbone in single phase
            
            # Model configuration - train from scratch
            pretrained=False  # Build models from scratch without pretrained weights
        )
        
        # Process results
        if result.get('success'):
            print("\n" + "=" * 80)
            print(f"🎉 {args.backbone.upper()} Training Completed!")
            print("=" * 80)
            
            # Display results
            pipeline_summary = result.get('pipeline_summary', {})
            total_duration = pipeline_summary.get('total_duration', 0)
            
            print(f"📊 Training Summary:")
            print(f"   • Backbone: {args.backbone}")
            print(f"   • Epochs: {args.epochs}")
            print(f"   • Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
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
            
            print(f"\n✅ Memory optimization handled automatically by core services")
            return 0
            
        else:
            print("\n" + "=" * 80)
            print("❌ Training Failed")
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