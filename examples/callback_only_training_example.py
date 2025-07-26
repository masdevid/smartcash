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
    """Create a metrics callback that prints training metrics."""
    def metrics_callback(phase: str, epoch: int, metrics: dict, **kwargs):
        """Handle metrics updates from training."""
        print(f"📊 METRICS [{phase.upper()}] Epoch {epoch}:")
        
        if metrics:
            # Print loss metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric_name}: {value:.4f}")
                else:
                    print(f"    {metric_name}: {value}")
        
        # Print additional info from kwargs
        if verbose and kwargs:
            for key, value in kwargs.items():
                if key not in ['phase', 'epoch', 'metrics']:
                    print(f"    {key}: {value}")
    
    return metrics_callback


def create_progress_callback(use_tqdm: bool = True, verbose: bool = True):
    """Create a triple tqdm progress callback for visual training progress."""
    
    # Storage for progress bars
    progress_bars = {}
    
    def _format_phase_display(phase: str) -> str:
        """Format phase name for display."""
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
    
    def _handle_training_phase_progress(phase: str, current: int, total: int, message: str, **kwargs):
        """Handle training phase progress with triple tqdm bars."""
        if not use_tqdm:
            # Fallback to simple text
            percentage = (current / total) * 100 if total > 0 else 0
            phase_display = _format_phase_display(phase)
            print(f"🔄 PROGRESS [{phase_display}] {percentage:.0f}% ({current}/{total}): {message}")
            return
        
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
            batch_idx = kwargs.get('batch_idx', 0)
            batch_total = kwargs.get('batch_total', 1)
            metrics = kwargs.get('metrics', {})
            phase_num = _get_phase_number(phase)
            
            # Level 1: Overall Training Progress Bar (position 0)
            overall_bar_key = "overall_training"
            if overall_bar_key not in progress_bars:
                # Calculate total steps across all training phases
                total_steps = total  # This is the total epochs for current phase
                progress_bars[overall_bar_key] = tqdm(
                    total=100, desc="🚀 Overall Training", unit="%", 
                    position=0, leave=True, colour='blue'
                )
            
            # Level 2: Phase Epoch Progress Bar (position 1)
            epoch_bar_key = f"{phase}_epoch"
            if epoch_bar_key not in progress_bars:
                progress_bars[epoch_bar_key] = tqdm(
                    total=total, desc=f"📚 Phase {phase_num} Epochs", unit="epoch", 
                    position=1, leave=True, colour='green'
                )
            
            # Level 3: Batch Progress Bar (position 2)
            batch_bar_key = f"{phase}_batch_{epoch}"
            if batch_bar_key not in progress_bars and batch_total > 1:
                progress_bars[batch_bar_key] = tqdm(
                    total=batch_total, desc=f"⚡ Phase {phase_num} Epoch {epoch} Batches", 
                    unit="batch", position=2, leave=False, colour='yellow'
                )
            
            # Update batch progress if available
            if batch_bar_key in progress_bars and batch_idx >= 0:
                batch_bar = progress_bars[batch_bar_key]
                postfix = {}
                
                if metrics:
                    loss = metrics.get('train_loss', 0)
                    layer1_acc = metrics.get('layer_1_accuracy', 0)
                    layer1_f1 = metrics.get('layer_1_f1', 0)
                    
                    if loss > 0: postfix['Loss'] = f"{loss:.4f}"
                    if layer1_acc > 0: postfix['L1_Acc'] = f"{layer1_acc:.3f}"
                    if layer1_f1 > 0: postfix['L1_F1'] = f"{layer1_f1:.3f}"
                
                batch_bar.set_postfix(postfix)
                batch_bar.n = batch_idx
                batch_bar.refresh()
            
            # Update epoch progress when epoch completes
            if current == total:
                # Close batch bar for this epoch
                if batch_bar_key in progress_bars:
                    progress_bars[batch_bar_key].close()
                    del progress_bars[batch_bar_key]
                
                # Update epoch bar
                epoch_bar = progress_bars[epoch_bar_key]
                epoch_postfix = {}
                if metrics:
                    loss = metrics.get('train_loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    
                    epoch_postfix.update({
                        'Train_Loss': f"{loss:.4f}",
                        'Val_Loss': f"{val_loss:.4f}"
                    })
                
                epoch_bar.set_postfix(epoch_postfix)
                epoch_bar.update(1)
                
                # Close epoch bar when training phase completes
                if current >= total:
                    epoch_bar.close()
                    del progress_bars[epoch_bar_key]
                    
                    # Update overall progress
                    overall_bar = progress_bars.get(overall_bar_key)
                    if overall_bar:
                        # Estimate overall progress (this is approximate)
                        if phase == 'training_phase_1':
                            overall_progress = 60  # 60% when phase 1 completes
                        elif phase == 'training_phase_2':
                            overall_progress = 90  # 90% when phase 2 completes
                        elif phase == 'training_phase_single':
                            overall_progress = 90  # 90% when single phase completes
                        else:
                            overall_progress = min(95, overall_bar.n + 10)
                        
                        overall_bar.n = overall_progress
                        overall_bar.set_description(f"🚀 Overall Training - {phase_display} Complete")
                        overall_bar.refresh()
        else:
            # Handle non-training phase progress
            _handle_phase_progress(phase, current, total, message)
    
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
        
        # Map phases to overall progress ranges
        phase_progress_map = {
            'preparation': (0, 10),
            'build_model': (10, 30),
            'validate_model': (30, 50),
            'training_phase_1': (50, 70),
            'training_phase_2': (70, 90),
            'training_phase_single': (50, 90),
            'summary_visualization': (90, 100)
        }
        
        if phase in phase_progress_map:
            start, end = phase_progress_map[phase]
            overall_progress = start + (percentage / 100) * (end - start)
            overall_bar.n = int(overall_progress)
            overall_bar.set_description(f"🚀 Overall Training - {phase_display}")
            overall_bar.refresh()
        
        # Handle phase completion
        if percentage >= 100:
            if phase == 'summary_visualization':
                # Training is complete
                overall_bar.n = 100
                overall_bar.set_description("🚀 Training Complete!")
                overall_bar.refresh()
                overall_bar.close()
                del progress_bars[overall_bar_key]
    
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        """Main progress callback with triple tqdm support."""
        phase_display = _format_phase_display(phase)
        
        if phase in ['training_phase_1', 'training_phase_2', 'training_phase_single']:
            _handle_training_phase_progress(phase, current, total, message, **kwargs)
        else:
            _handle_phase_progress(phase, current, total, message)
    
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
        
        # Get training arguments and add callbacks
        training_kwargs = get_training_kwargs(args)
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