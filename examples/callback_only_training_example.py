#!/usr/bin/env python3
"""
Callback-Only Training Example

This example demonstrates a minimal training setup that only uses callbacks 
to display training information. No other output is shown except what comes 
from log_callback, metrics_callback, and progress_callback.

Usage:
    python examples/callback_only_training_example.py --backbone cspdarknet --phase1-epochs 1 --verbose
    python examples/callback_only_training_example.py --training-mode single_phase --phase1-epochs 2 --force-cpu
    python examples/callback_only_training_example.py --optimizer adamw --scheduler cosine --weight-decay 1e-2 --phase1-epochs 1
    python examples/callback_only_training_example.py --resume data/checkpoints/best_model.pt --resume-optimizer --resume-scheduler
    
Validation Metrics:
    # Training automatically uses hierarchical validation (YOLOv5 + per-layer metrics)
    # No additional flags needed - optimized for Indonesian banknote detection
    
Features:
    - Automatic memory cleanup on interruption (Ctrl+C)
    - Real-time memory monitoring in verbose mode
    - Graceful signal handling for clean exits
    - GPU/MPS cache clearing and garbage collection
"""

# Fix OpenMP duplicate library issue before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import gc
import torch
import psutil
import signal
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, Any, Optional

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import run_full_training_pipeline
from examples.training_args_helper import create_training_arg_parser, get_training_kwargs
from smartcash.model.training.utils.metric_color_utils import ColorScheme
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback


def load_legacy_checkpoint_for_resume(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint for resume training.
    
    This uses the core checkpoint utilities for consistent behavior across the system.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Resume information dictionary or None if failed
    """
    # Import core checkpoint utilities
    from smartcash.model.training.utils.checkpoint_utils import load_checkpoint_for_resume
    
    # Use the core function with verbose output for user feedback
    return load_checkpoint_for_resume(checkpoint_path, verbose=True)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory(verbose: bool = False):
    """
    Comprehensive memory cleanup function.
    
    Args:
        verbose: Whether to print memory cleanup details
    """
    if verbose:
        memory_before = get_memory_usage()
        print(f"🧹 MEMORY CLEANUP: Starting cleanup (Memory: {memory_before:.1f} MB)")
    
    # Clear PyTorch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if verbose:
            print("   ✅ Cleared CUDA cache")
    
    # Clear MPS cache if available (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            if verbose:
                print("   ✅ Cleared MPS cache")
        except Exception:
            pass  # Ignore if MPS cache clearing fails
    
    # Force garbage collection
    collected = gc.collect()
    if verbose:
        print(f"   ✅ Garbage collection: {collected} objects collected")
    
    # Additional aggressive cleanup for large models
    for _ in range(3):
        gc.collect()
    
    if verbose:
        memory_after = get_memory_usage()
        memory_freed = memory_before - memory_after
        print(f"   ✅ Memory cleanup complete (Memory: {memory_after:.1f} MB, Freed: {memory_freed:.1f} MB)")


def setup_signal_handlers(verbose: bool = False):
    """
    Setup signal handlers for graceful interruption handling.
    
    Args:
        verbose: Whether to print signal handler setup details
    """
    def signal_handler(signum, _):
        """Handle interruption signals with memory cleanup."""
        signal_name = signal.Signals(signum).name
        print(f"\n🛑 TRAINING INTERRUPTED by {signal_name}")
        print("🧹 Performing emergency memory cleanup...")
        
        try:
            cleanup_memory(verbose=verbose)
            print("✅ Memory cleanup completed successfully")
        except Exception as e:
            print(f"⚠️ Memory cleanup error: {str(e)}")
        
        print("👋 Exiting gracefully...")
        sys.exit(1)
    
    # Register signal handlers for common interruption signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    if verbose:
        print("🛡️ Signal handlers registered for graceful interruption handling")


def create_log_callback(verbose: bool = True):
    """Create a log callback that prints all log messages with enhanced phase and epoch information."""
    def log_callback(level: str, message: str, data: dict = None):
        """Handle log messages from the training pipeline."""
        # Format level with appropriate emoji
        level_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'debug': '🔍',
            'critical': '🚨'
        }
        
        icon = level_icons.get(level.lower(), '📝')
        
        # Enhance message with phase and epoch context if available
        enhanced_message = message
        if data:
            # Add phase information if available
            if 'phase' in data:
                phase_display = f"Phase {data['phase']}"
                enhanced_message = f"[{phase_display}] {message}"
            
            # Add epoch information if available
            if 'epoch' in data:
                epoch_info = f"Epoch {data['epoch']}"
                # If message doesn't already contain epoch info
                if 'epoch' not in message.lower() and 'e' not in message[:10].lower():
                    enhanced_message = f"[{epoch_info}] {enhanced_message}"
            
            # Add resume context if this is a resumed training log
            if 'resumed_from' in data:
                enhanced_message = f"[RESUMED from E{data['resumed_from']}] {enhanced_message}"
        
        print(f"{icon} [{level.upper()}] {enhanced_message}")
        
        # Print additional data if available and verbose mode is on
        if verbose and data:
            for key, value in data.items():
                if key not in ['message', 'phase', 'epoch', 'resumed_from']:  # Skip already processed keys
                    print(f"    {key}: {value}")
    
    return log_callback


def create_metrics_callback(verbose: bool = True):
    """Create an enhanced metrics callback with UI color support."""
    return create_ui_metrics_callback(
        verbose=verbose,
        console_scheme=ColorScheme.EMOJI,
        ui_callback=None  # Can be set later for UI integration
    )


def create_progress_callback(use_tqdm: bool = True, verbose: bool = True):
    """Create a triple tqdm progress callback for the new 3-level progress system with memory monitoring."""
    
    # Storage for progress bars
    progress_bars = {}
    last_memory_check = [0]  # Use list to make it mutable in nested functions
    
    def _format_phase_display(phase: str) -> str:
        """Format phase name for display."""
        if phase == 'overall':
            return 'Overall Training'
        elif phase == 'epoch':
            return 'Epoch Progress'
        elif phase == 'batch':
            return 'Batch Progress'
        else:
            return phase.replace('_', ' ').title()
    
    
    # Old training phase progress handling removed - now using 3-level system
    
    def _handle_phase_progress(phase: str, current: int, total: int, message: str):
        """Handle non-training phase progress."""
        if not use_tqdm:
            percentage = (current / total) * 100 if total > 0 else 0
            phase_display = _format_phase_display(phase)
            print(f"🔄 PROGRESS [{phase_display}] {percentage:.0f}% ({current}/{total}): {message}")
            return
        
        phase_display = _format_phase_display(phase)
        
        # Update overall progress bar
        overall_bar_key = "overall_training"
        if overall_bar_key not in progress_bars:
            progress_bars[overall_bar_key] = tqdm(
                total=100, desc="🚀 Overall Training", unit="%", 
                position=0, leave=True, colour='blue'
            )
        
        overall_bar = progress_bars[overall_bar_key]
        percentage = (current / total) * 100 if total > 0 else 0
        
        # Map phases to overall progress ranges - updated for new phase names
        phase_progress_map = {
            'preparation': (0, 10),
            'build_model': (10, 30),
            'validate_model': (30, 50),
            'training_phase_1': (50, 70),
            'training_phase_2': (70, 90),
            'training_phase_single': (50, 90),
            'finalize': (90, 100)  # Updated from 'summary_visualization'
        }
        
        if phase in phase_progress_map:
            start, end = phase_progress_map[phase]
            overall_progress = start + (percentage / 100) * (end - start)
            
            # Handle finalize phase completion/failure properly
            if phase == 'finalize':
                if percentage >= 99:  # Account for both success (100%) and failure (99%)
                    if "failed" in message.lower() or "❌" in message:
                        # Failed finalize - stop at 95% to indicate incomplete
                        overall_progress = 95
                        overall_bar.set_description("🚀 Training Failed - See logs for details")
                    else:
                        # Successful finalize - reach 100%
                        overall_progress = 100
                        overall_bar.set_description("🚀 Training Complete!")
                else:
                    overall_bar.set_description(f"🚀 Overall Training - {phase_display}")
            else:
                overall_bar.set_description(f"🚀 Overall Training - {phase_display}")
            
            overall_bar.n = int(overall_progress)
            overall_bar.refresh()
        
        # Handle phase completion
        if percentage >= 99:  # Handle both 100% (success) and 99% (failure)
            if phase == 'finalize':  # Updated from 'summary_visualization'
                # Training phase is done - handle success or failure
                if "failed" in message.lower() or "❌" in message:
                    # Failed training - stop at 95%
                    overall_bar.n = 95
                    overall_bar.set_description("🚀 Training Failed - See logs for details")
                else:
                    # Successful training - reach 100%
                    overall_bar.n = 100
                    overall_bar.set_description("🚀 Training Complete!")
                
                overall_bar.refresh()
                overall_bar.close()
                del progress_bars[overall_bar_key]
    
    def progress_callback(progress_type: str, current: int, total: int, message: str = "", **kwargs):
        """
        Main progress callback with 3-level progress system:
        1. 'overall' - Overall pipeline progress (5-6 phases)
        2. 'epoch' - Epoch progress within training phases  
        3. 'batch' - Batch progress within epochs
        """
        if not use_tqdm:
            # Fallback to simple text output
            percentage = (current / total) * 100 if total > 0 else 0
            display_name = _format_phase_display(progress_type)
            print(f"🔄 PROGRESS [{display_name}] {percentage:.0f}% ({current}/{total}): {message}")
            return
        
        # Handle the 3-level progress system
        if progress_type == 'overall':
            _handle_overall_progress(current, total, message, **kwargs)
        elif progress_type == 'epoch':
            _handle_epoch_progress(current, total, message, **kwargs)
        elif progress_type == 'batch':
            _handle_batch_progress(current, total, message, **kwargs)
        else:
            # Legacy support for old phase-based progress
            _handle_phase_progress(progress_type, current, total, message)
    
    def _handle_overall_progress(current: int, total: int, message: str, **_kwargs):
        """Handle overall pipeline progress (Level 1)."""
        bar_key = "overall"
        if bar_key not in progress_bars:
            progress_bars[bar_key] = tqdm(
                total=total, desc="🚀 Overall Training", unit="%", 
                position=0, leave=True, colour='blue'
            )
        
        bar = progress_bars[bar_key]
        bar.n = current
        
        # Update description - use completion message if at 100%
        if current >= total:
            bar.set_description("🚀 Training Complete!")
        else:
            bar.set_description(f"🚀 Overall Training - {message}")
        bar.refresh()
        
        # Close when complete
        if current >= total:
            bar.close()
            del progress_bars[bar_key]
    
    def _handle_epoch_progress(current: int, total: int, message: str, **kwargs):
        """Handle epoch progress (Level 2)."""
        bar_key = "epoch"
        if bar_key not in progress_bars:
            progress_bars[bar_key] = tqdm(
                total=total, desc="📚 Epoch Progress", unit="epoch", 
                position=1, leave=True, colour='green'
            )
        
        bar = progress_bars[bar_key]
        bar.n = current
        
        # Use actual epoch number from kwargs if available (for resumed training)
        actual_epoch = kwargs.get('epoch', current)
        bar.set_description(f"📚 Epoch Progress - Epoch {actual_epoch}")
        
        # Add postfix with additional info
        postfix = {}
        if 'phase' in kwargs:
            postfix['Phase'] = kwargs['phase']
        if actual_epoch != current:
            postfix['Resumed'] = f"From E{actual_epoch}"
        bar.set_postfix(postfix)
        bar.refresh()
        
        # Close when complete or early stopping
        if current >= total or "Early stopping" in message or "100%" in message:
            bar.close()
            del progress_bars[bar_key]
    
    def _handle_batch_progress(current: int, total: int, message: str, **kwargs):
        """Handle batch progress (Level 3) with memory monitoring."""
        bar_key = "batch"
        if bar_key not in progress_bars:
            progress_bars[bar_key] = tqdm(
                total=total, desc="⚡ Batch Progress", unit="batch", 
                position=2, leave=False, colour='yellow'
            )
        
        bar = progress_bars[bar_key]
        bar.n = current
        bar.set_description(f"⚡ Batch Progress - {message}")
        
        # Add loss and other metrics to postfix
        postfix = {}
        if 'loss' in kwargs:
            postfix['Loss'] = f"{kwargs['loss']:.4f}"
        if 'epoch' in kwargs:
            postfix['Epoch'] = kwargs['epoch']
        
        # Add memory monitoring every 10 batches
        if verbose and current % 10 == 0:
            try:
                current_memory = get_memory_usage()
                postfix['Mem'] = f"{current_memory:.0f}MB"
                
                # Check for significant memory increase (> 500MB from last check)
                if last_memory_check[0] > 0 and current_memory - last_memory_check[0] > 500:
                    # Perform light cleanup
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                
                last_memory_check[0] = current_memory
            except Exception:
                pass  # Ignore memory monitoring errors
        
        bar.set_postfix(postfix)
        bar.refresh()
        
        # Close when complete
        if current >= total:
            bar.close()
            del progress_bars[bar_key]
    
    return progress_callback


def main():
    """Main function for callback-only training demonstration."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = create_training_arg_parser('SmartCash Callback-Only Training Example')
    args = parser.parse_args()
    
    print("🚀 STARTING SmartCash Callback-Only Training")
    print("=" * 60)
    
    # Setup memory management and signal handlers
    setup_signal_handlers(verbose=args.verbose)
    if args.verbose:
        initial_memory = get_memory_usage()
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    print("Only callback outputs will be shown below:")
    print("=" * 60)
    
    try:
        # Create callbacks - these are the ONLY output sources
        log_callback = create_log_callback(args.verbose)
        metrics_callback = create_metrics_callback(args.verbose)
        progress_callback = create_progress_callback(args.verbose)
        
        # Show training configuration info
        if args.training_mode == 'two_phase':
            print("🎯 TWO-PHASE MODE: Early stopping disabled for Phase 1, enabled for Phase 2")
        else:
            print("🎯 SINGLE-PHASE MODE: Early stopping uses your configuration")
        
        # Show optimizer and scheduler info
        print(f"⚙️ OPTIMIZER: {args.optimizer.upper()} with {args.scheduler} scheduler (weight_decay={args.weight_decay})")
        if args.scheduler == 'cosine':
            print(f"   └─ Cosine annealing: eta_min={args.cosine_eta_min}")
        
        # Show validation metrics configuration
        print("📊 VALIDATION METRICS: YOLOv5 hierarchical + per-layer metrics")
        print("   └─ YOLOv5: accuracy=mAP@0.5, precision/recall/F1 from hierarchical object detection")
        print("   └─ Per-layer: layer_1_accuracy, layer_2_accuracy, layer_3_accuracy")
        
        # Show resume info
        if args.resume:
            if args.resume == 'auto':
                print("🔄 RESUME TRAINING: Auto-detecting latest checkpoint")
            else:
                print(f"🔄 RESUME TRAINING: Loading from {args.resume}")
            resume_components = []
            if args.resume_optimizer:
                resume_components.append("optimizer state")
            if args.resume_scheduler:
                resume_components.append("scheduler state")
            if resume_components:
                print(f"   └─ Resuming: {', '.join(resume_components)}")
            if args.resume_epoch:
                print(f"   └─ Epoch override: {args.resume_epoch}")
        else:
            print("🔄 TRAINING FROM SCRATCH: No checkpoint resume")
        
        # Get training arguments
        training_kwargs = get_training_kwargs(args)
        
        # Set checkpoint directory to be relative to project root
        training_kwargs['checkpoint_dir'] = str(Path(project_root) / 'data' / 'checkpoints')
        
        # Handle resume from existing checkpoint format
        if args.resume:
            checkpoint_path = args.resume
            
            # Auto-detect latest checkpoint if --resume was used without path
            if args.resume == 'auto':
                from smartcash.model.training.utils.checkpoint_utils import find_latest_checkpoint
                checkpoint_path = find_latest_checkpoint(args.checkpoint_dir, args.backbone)
                
                if not checkpoint_path:
                    print("❌ No 'last_*.pt' checkpoints found for auto-resume")
                    print(f"💡 Searched in: {args.checkpoint_dir}")
                    print("💡 Run training without --resume to start fresh, or specify a checkpoint path")
                    return 1
                
                print(f"🔍 Auto-detected checkpoint: {Path(checkpoint_path).name}")
            
            # Load the checkpoint
            resume_info = load_legacy_checkpoint_for_resume(checkpoint_path)
            if resume_info:
                training_kwargs.update({
                    'resume_from_checkpoint': True,
                    'resume_info': resume_info
                })
                print(f"✅ Successfully loaded checkpoint: epoch {resume_info['epoch']} (phase {resume_info.get('phase', 'N/A')})")
            else:
                print(f"❌ FAILED to load checkpoint {checkpoint_path}")
                print("❌ TRAINING TERMINATED - Invalid checkpoint file")
                print("💡 Please check the checkpoint path and try again")
                return 1  # Exit with error code instead of continuing
        
        # Set model configuration using arguments from args
        # Use enhanced model builder with YOLOv5 integration
        training_kwargs['model'] = {
            'model_name': 'smartcash_yolov5_integrated',
            'backbone': args.backbone,
            'pretrained': args.pretrained,
            'layer_mode': 'multi' if args.training_mode == 'two_phase' else args.single_layer_mode,
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
            'num_classes': 7,  # Fixed for banknote detection
            'img_size': 640,   # Standard size for YOLO models
            'feature_optimization': {'enabled': True},
        }
        
        # Create callbacks
        training_kwargs.update({
            'log_callback': log_callback,
            'metrics_callback': metrics_callback,
            'progress_callback': progress_callback,
            'verbose': args.verbose
        })
        
        # Run training pipeline - all output comes from callbacks
        result = run_full_training_pipeline(**training_kwargs)
        
        # Only show final result status
        print("=" * 60)
        if result.get('success'):
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
            exit_code = 0
        else:
            print("❌ TRAINING FAILED")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
            exit_code = 1
        print("=" * 60)
        
        # Perform final memory cleanup
        if args.verbose:
            print("🧹 Performing final memory cleanup...")
        cleanup_memory(verbose=args.verbose)
        
        return exit_code
            
    except KeyboardInterrupt:
        print("\n⚠️ TRAINING INTERRUPTED BY USER")
        print("🧹 Performing cleanup after interruption...")
        cleanup_memory(verbose=args.verbose)
        return 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("🧹 Performing cleanup after error...")
        cleanup_memory(verbose=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())