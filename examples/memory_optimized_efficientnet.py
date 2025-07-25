#!/usr/bin/env python3
"""
Memory-optimized backbone training with gradient accumulation and mixed precision
Solves both MPS memory limitations and CPU semaphore leaking issues
Supports EfficientNet-B4 and CSPDarkNet (YOLOv5s) backbones with tqdm progress tracking
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

class MemoryOptimizer:
    """Memory optimization utilities for large model training"""
    
    @staticmethod
    def setup_memory_efficient_device():
        """Setup the most memory-efficient device available"""
        if torch.backends.mps.is_available():
            # MPS with memory optimization
            try:
                torch.mps.empty_cache()
                device = 'mps'
                print("🚀 Using MPS with memory optimization")
            except RuntimeError as e:
                print(f"⚠️ MPS cache error: {e}")
                device = 'mps'
                print("🚀 Using MPS (cache warning ignored)")
        elif torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.empty_cache()
            print("🚀 Using CUDA with memory optimization")
        else:
            device = 'cpu'
            # Optimize CPU threading to prevent semaphore leaking
            torch.set_num_threads(min(4, os.cpu_count() or 4))
            print(f"🖥️ Using CPU with {torch.get_num_threads()} threads")
        
        return device
    
    @staticmethod
    def optimize_memory_settings():
        """Configure PyTorch for memory efficiency"""
        # Enable memory efficient attention if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Aggressive MPS memory management
        if torch.backends.mps.is_available():
            # Remove any existing memory ratio settings that might conflict
            if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
                del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
            print("🧠 MPS conservative memory management enabled")
        
        # CPU optimization to prevent semaphore leaking
        os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count() or 4))
        os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count() or 4))
        
        # PyTorch memory optimizations
        torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
        
        print("⚙️ Memory optimization settings applied")
    
    @staticmethod
    def cleanup_memory():
        """Force memory cleanup with aggressive MPS management"""
        # Force garbage collection
        gc.collect()
        
        if torch.backends.mps.is_available():
            try:
                # Multiple cache clearing attempts for MPS
                for _ in range(3):
                    torch.mps.empty_cache()
                    gc.collect()
                print("🧹 MPS memory cleaned")
            except RuntimeError as e:
                print(f"⚠️ MPS cache warning: {e}")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 CUDA memory cleaned")
        else:
            print("🧹 CPU memory cleaned")
    
    @staticmethod
    def emergency_memory_cleanup():
        """Emergency memory cleanup for out-of-memory situations"""
        print("🚨 Emergency memory cleanup initiated...")
        
        # Force multiple garbage collection passes
        for i in range(5):
            gc.collect()
        
        if torch.backends.mps.is_available():
            try:
                # Aggressive MPS cleanup
                for _ in range(5):
                    torch.mps.empty_cache()
                    gc.collect()
                print("🚨 Emergency MPS cleanup completed")
            except RuntimeError:
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        print("🚨 Emergency cleanup finished")

def create_tqdm_progress_callback():
    """Progress callback with tqdm progress bars and memory monitoring"""
    phase_bars = {}
    
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        # Memory monitoring
        if torch.backends.mps.is_available():
            memory_info = "MPS"
        elif torch.cuda.is_available():
            memory_info = f"GPU:{torch.cuda.memory_allocated()/1024**3:.1f}GB"
        else:
            memory_info = "CPU"
        
        # Create or update tqdm bar for this phase
        if phase not in phase_bars:
            phase_display = phase.replace('_', ' ').title()
            phase_bars[phase] = tqdm(
                total=total,
                desc=f"🔄 {phase_display} [{memory_info}]",
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
            bar.set_description(f"🔄 {phase_display} [{memory_info}]: {message}")
        
        # Handle training phase specifics
        if phase in ['training_phase_1', 'training_phase_2']:
            if 'epoch' in kwargs:
                epoch = kwargs['epoch']
                metrics = kwargs.get('metrics', {})
                phase_num = "1" if phase == 'training_phase_1' else "2"
                
                # Update progress bar description with current epoch
                bar.set_description(f"🔄 Phase {phase_num} - Epoch {epoch}/{kwargs.get('total_epochs', '?')} [{memory_info}]")
                
                if current == total and metrics:
                    # Close the progress bar and print detailed epoch results
                    bar.close()
                    
                    tqdm.write("\n" + "="*60)
                    tqdm.write(f"📊 PHASE {phase_num} - EPOCH {epoch} COMPLETED [{memory_info}]")
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
                    
                    # Force memory cleanup after each epoch
                    MemoryOptimizer.cleanup_memory()
                    
                    # Additional aggressive cleanup for MPS
                    if torch.backends.mps.is_available():
                        MemoryOptimizer.emergency_memory_cleanup()
                    
                    # Remove from tracking
                    del phase_bars[phase]
                else:
                    # Update progress bar with batch info if available
                    batch_idx = kwargs.get('batch_idx', 0)
                    total_batches = kwargs.get('total_batches', 0)
                    if batch_idx > 0 and total_batches > 0:
                        bar.set_description(f"🔄 Phase {phase_num} - Epoch {epoch} - Batch {batch_idx}/{total_batches} [{memory_info}]")
        
        # Complete phase handling
        if current >= total:
            bar.close()
            phase_display = phase.replace('_', ' ').title()
            tqdm.write(f"✅ {phase_display} [{memory_info}]: Completed")
            if phase in phase_bars:
                del phase_bars[phase]
    
    return progress_callback

def parse_arguments():
    """Parse command line arguments for backbone configuration"""
    parser = argparse.ArgumentParser(
        description="Memory-optimized training for EfficientNet-B4 and CSPDarkNet (YOLOv5s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Backbone selection
    parser.add_argument(
        '--backbone', '-b',
        type=str,
        default='efficientnet_b4',
        choices=['efficientnet_b4', 'cspdarknet'],
        help='Backbone model to use for training (EfficientNet-B4 or CSPDarkNet YOLOv5s backbone)'
    )
    
    # Training configuration
    parser.add_argument('--phase1-epochs', type=int, default=2, help='Number of epochs for phase 1 (frozen backbone)')
    parser.add_argument('--phase2-epochs', type=int, default=3, help='Number of epochs for phase 2 (fine-tuning)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override default batch size')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--disable-tqdm', action='store_true', help='Disable tqdm progress bars')
    
    # Memory optimization
    parser.add_argument('--gradient-accumulation', type=int, default=None, help='Override gradient accumulation steps')
    parser.add_argument('--mixed-precision', action='store_true', help='Force enable mixed precision (if supported)')
    
    return parser.parse_args()

def get_model_memory_config(backbone: str, device: str, args):
    """Get memory configuration based on backbone and device"""
    # Ultra-conservative configurations to prevent MPS OOM
    backbone_configs = {
        'efficientnet_b4': {'mps_batch': 1, 'cuda_batch': 4, 'cpu_batch': 1},  # Reduced MPS batch
        'cspdarknet': {'mps_batch': 1, 'cuda_batch': 6, 'cpu_batch': 1},       # Reduced MPS batch
    }
    
    config = backbone_configs.get(backbone, backbone_configs['efficientnet_b4'])
    
    # Select device-specific configuration
    if device == 'mps':
        batch_size = config['mps_batch']
        gradient_accumulation_steps = 8 // batch_size  # Target effective batch size of 8
        use_mixed_precision = False  # MPS doesn't support AMP yet
    elif device == 'cuda':
        batch_size = config['cuda_batch']
        gradient_accumulation_steps = max(1, 8 // batch_size)  # Target effective batch size of 8
        use_mixed_precision = True
    else:  # CPU
        batch_size = config['cpu_batch']
        gradient_accumulation_steps = 8 // batch_size  # Target effective batch size of 8
        use_mixed_precision = False
    
    # Apply user overrides
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        gradient_accumulation_steps = args.gradient_accumulation
    if args.mixed_precision:
        use_mixed_precision = True
    
    return batch_size, gradient_accumulation_steps, use_mixed_precision

def main():
    """Run memory-optimized backbone training with configurable models"""
    args = parse_arguments()
    
    print("=" * 80)
    print(f"🧠 SmartCash Memory-Optimized Training - {args.backbone.upper()}")
    print("=" * 80)
    
    # Setup memory optimization
    MemoryOptimizer.optimize_memory_settings()
    device = MemoryOptimizer.setup_memory_efficient_device()
    
    if args.force_cpu:
        device = 'cpu'
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        print("🖥️ Forced CPU mode with thread limiting")
    
    # Get memory configuration for the selected backbone
    batch_size, gradient_accumulation_steps, use_mixed_precision = get_model_memory_config(
        args.backbone, device, args
    )
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    print(f"🎯 {device.upper()} Configuration:")
    print(f"   • Backbone: {args.backbone}")
    print(f"   • Batch size: {batch_size} (with {gradient_accumulation_steps}x accumulation = effective batch size {effective_batch_size})")
    print(f"   • Mixed precision: {'Enabled' if use_mixed_precision else 'Disabled'}")
    print(f"   • Phase 1 epochs: {args.phase1_epochs} (frozen backbone)")
    print(f"   • Phase 2 epochs: {args.phase2_epochs} (fine-tuning)")
    print(f"   • Progress bars: {'Disabled' if args.disable_tqdm else 'tqdm enabled'}")
    print(f"   • Device: {device}")
    print("=" * 80)
    
    try:
        # Create progress callback
        if args.disable_tqdm:
            progress_callback = None
        else:
            progress_callback = create_tqdm_progress_callback()
        
        tqdm.write("🚀 Starting memory-optimized training pipeline...")
        if device == 'cpu':
            tqdm.write("⚠️  CPU mode with semaphore leak prevention enabled")
        tqdm.write("")
        
        # Clean memory before starting and apply emergency cleanup for MPS
        MemoryOptimizer.cleanup_memory()
        if device == 'mps':
            MemoryOptimizer.emergency_memory_cleanup()
            tqdm.write("🚨 Pre-training memory cleanup completed for MPS")
        
        result = run_full_training_pipeline(
            backbone=args.backbone,
            phase_1_epochs=args.phase1_epochs,
            phase_2_epochs=args.phase2_epochs,
            checkpoint_dir='data/checkpoints',
            progress_callback=progress_callback,
            verbose=True,
            force_cpu=(device == 'cpu'),
            training_mode='two_phase',
            
            # Memory optimization settings
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision,
            
            # Training configuration
            loss_type='uncertainty_multi_task',
            head_lr_p1=0.001,
            head_lr_p2=0.0001,
            backbone_lr=1e-5,
            
            # Memory-friendly early stopping
            early_stopping_enabled=True,
            early_stopping_patience=10,  # Reduced patience for faster convergence
            early_stopping_metric='val_map50',
            early_stopping_mode='max',
            early_stopping_min_delta=0.002,
            
            # Additional memory optimizations
            dataloader_num_workers=0,  # Always 0 to prevent semaphore leaks
            pin_memory=False,  # Always False to prevent memory issues
            persistent_workers=False,  # Reduce memory footprint
            
            # Gradient management
            max_grad_norm=1.0,  # Gradient clipping
            weight_decay=0.0005
        )
        
        # Final memory cleanup
        MemoryOptimizer.cleanup_memory()
        if device == 'mps':
            MemoryOptimizer.emergency_memory_cleanup()
        
        # Process results
        if result.get('success'):
            tqdm.write("\n" + "=" * 80)
            tqdm.write(f"🎉 Memory-Optimized {args.backbone.upper()} Training Completed!")
            tqdm.write("=" * 80)
            
            # Display results
            pipeline_summary = result.get('pipeline_summary', {})
            total_duration = pipeline_summary.get('total_duration', 0)
            
            tqdm.write(f"📊 Training Summary:")
            tqdm.write(f"   • Backbone: {args.backbone}")
            tqdm.write(f"   • Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            tqdm.write(f"   • Device used: {device}")
            tqdm.write(f"   • Effective batch size: {effective_batch_size}")
            tqdm.write(f"   • Gradient accumulation: {gradient_accumulation_steps} steps")
            tqdm.write(f"   • Mixed precision: {'Enabled' if use_mixed_precision else 'Disabled'}")
            tqdm.write(f"   • Phases completed: {pipeline_summary.get('phases_completed', 0)}/6")
            
            # Training results
            training_result = result.get('final_training_result', {})
            if training_result.get('success'):
                best_metrics = training_result.get('best_metrics', {})
                
                tqdm.write(f"\n🏆 Training Results:")
                tqdm.write(f"   • Final train loss: {best_metrics.get('train_loss', 0):.4f}")
                tqdm.write(f"   • Final val loss: {best_metrics.get('val_loss', 0):.4f}")
                
                # Layer metrics
                tqdm.write(f"\n📊 Layer Performance:")
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    acc = best_metrics.get(f'{layer}_accuracy', 0)
                    if acc > 0:
                        f1 = best_metrics.get(f'{layer}_f1', 0)
                        tqdm.write(f"   • {layer.upper()}: Acc={acc:.4f} F1={f1:.4f}")
            
            tqdm.write(f"\n💡 Memory Optimization Results:")
            if device == 'cpu':
                tqdm.write("   • ✅ CPU semaphore leaking prevented with thread limiting")
            elif device == 'mps':
                tqdm.write("   • ✅ MPS memory fragmentation avoided with small batches + accumulation")
            else:
                tqdm.write("   • ✅ CUDA memory optimized with mixed precision")
            
            return 0
            
        else:
            tqdm.write("\n" + "=" * 80)
            tqdm.write("❌ Memory-Optimized Training Failed")
            tqdm.write("=" * 80)
            error_msg = result.get('error', 'Unknown error')
            tqdm.write(f"Error: {error_msg}")
            return 1
            
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Training interrupted by user")
        MemoryOptimizer.cleanup_memory()
        if device == 'mps':
            MemoryOptimizer.emergency_memory_cleanup()
        return 1
        
    except Exception as e:
        tqdm.write(f"\n❌ Unexpected error: {str(e)}")
        
        # Check if it's a memory error
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            tqdm.write("🚨 Memory error detected - running emergency cleanup")
            MemoryOptimizer.emergency_memory_cleanup()
            tqdm.write("💡 Try reducing batch size with --batch-size 1 or use --force-cpu")
        
        import traceback
        traceback.print_exc()
        MemoryOptimizer.cleanup_memory()
        return 1

if __name__ == "__main__":
    sys.exit(main())