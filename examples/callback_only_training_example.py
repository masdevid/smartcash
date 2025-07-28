#!/usr/bin/env python3
"""
Callback-Only Training Example

This example demonstrates a minimal training setup that only uses callbacks 
to display training information. No other output is shown except what comes 
from log_callback, metrics_callback, and progress_callback.

Usage:
    python examples/callback_only_training_example.py --backbone cspdarknet --phase1-epochs 1 --verbose
    python examples/callback_only_training_example.py --training-mode single_phase --phase1-epochs 2 --force-cpu
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import run_full_training_pipeline
from training_args_helper import create_training_arg_parser, get_training_kwargs
from smartcash.model.training.utils.metric_color_utils import ColorScheme
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback


def create_log_callback(verbose: bool = True):
    """Create a log callback that prints all log messages."""
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
        print(f"{icon} [{level.upper()}] {message}")
        
        # Print additional data if available and verbose mode is on
        if verbose and data:
            for key, value in data.items():
                if key != 'message':  # Avoid duplicate message
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
    """Create a triple tqdm progress callback for the new 3-level progress system."""
    
    # Storage for progress bars
    progress_bars = {}
    
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
    
    def _get_phase_number(phase: str) -> str:
        """Get phase number from phase name."""
        if phase == 'training_phase_1':
            return "1"
        elif phase == 'training_phase_2':
            return "2"
        elif phase == 'training_phase_single':
            return "Single"
        else:
            return "?"
    
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
    
    def _handle_overall_progress(current: int, total: int, message: str, **kwargs):
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
        bar.set_description(f"📚 Epoch Progress - {message}")
        
        # Add postfix with additional info
        postfix = {}
        if 'phase' in kwargs:
            postfix['Phase'] = kwargs['phase']
        bar.set_postfix(postfix)
        bar.refresh()
        
        # Close when complete or early stopping
        if current >= total or "Early stopping" in message or "100%" in message:
            bar.close()
            del progress_bars[bar_key]
    
    def _handle_batch_progress(current: int, total: int, message: str, **kwargs):
        """Handle batch progress (Level 3)."""
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
    print("Only callback outputs will be shown below:")
    print("=" * 60)
    
    try:
        # Create callbacks - these are the ONLY output sources
        log_callback = create_log_callback(args.verbose)
        metrics_callback = create_metrics_callback(args.verbose)
        progress_callback = create_progress_callback(args.verbose)
        
        # Show early stopping behavior info
        if args.training_mode == 'two_phase':
            print("🎯 TWO-PHASE MODE: Early stopping disabled for Phase 1, enabled for Phase 2")
        else:
            print("🎯 SINGLE-PHASE MODE: Early stopping uses your configuration")
        
        # Get training arguments
        training_kwargs = get_training_kwargs(args)
        
        # Set checkpoint directory to be relative to project root
        training_kwargs['checkpoint_dir'] = str(Path(project_root) / 'data' / 'checkpoints')
        
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
        else:
            print("❌ TRAINING FAILED")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
        print("=" * 60)
        
        return 0 if result.get('success') else 1
            
    except KeyboardInterrupt:
        print("\n⚠️ TRAINING INTERRUPTED BY USER")
        return 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())