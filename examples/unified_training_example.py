#!/usr/bin/env python3
"""
Example: Unified Training Pipeline

This example demonstrates the new run_full_training_pipeline API that merges
model building and training into a single, comprehensive workflow.

Key Features:
- 6-phase progress tracking: Preparation → Build Model → Validate Model → Training Phase 1 → Training Phase 2 → Summary & Visualization
- Platform-aware configuration (automatically optimized for Mac M1, Colab, Linux workstations, etc.)
- Configurable phase epochs (phase_1_epochs, phase_2_epochs)
- Unified checkpoint management under /data/checkpoints
- Automatic visualization generation
- Comprehensive progress callbacks with batch-level updates

Usage:
    python examples/unified_training_example.py --backbone cspdarknet --phase1-epochs 2 --phase2-epochs 3
    python examples/unified_training_example.py --backbone efficientnet_b4 --phase1-epochs 1 --phase2-epochs 1
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import run_full_training_pipeline

def _format_phase_display(phase: str) -> str:
    """Format phase name for display."""
    return phase.replace('_', ' ').title()

def _get_phase_number(phase: str) -> str:
    """Get phase number from phase name."""
    return "1" if phase == 'training_phase_1' else "2"

def _calculate_layer_averages(metrics: dict) -> tuple:
    """Calculate average layer metrics."""
    layer_accs = [metrics.get(f'layer_{i}_accuracy', 0) for i in [1, 2, 3]]
    layer_f1s = [metrics.get(f'layer_{i}_f1', 0) for i in [1, 2, 3]]
    
    avg_acc = sum(acc for acc in layer_accs if acc > 0) / max(1, len([acc for acc in layer_accs if acc > 0]))
    avg_f1 = sum(f1 for f1 in layer_f1s if f1 > 0) / max(1, len([f1 for f1 in layer_f1s if f1 > 0]))
    
    return avg_acc, avg_f1

def _print_layer_metrics(metrics: dict, phase_num: str, epoch: int, verbose: bool = True) -> None:
    """Print detailed layer metrics."""
    if not (verbose and metrics):
        return
        
    print(f"\n📊 Phase {phase_num} - Epoch {epoch} COMPLETED:")
    print(f"    Loss: Train={metrics.get('train_loss', 0):.4f} Val={metrics.get('val_loss', 0):.4f}")
    
    for layer in ['layer_1', 'layer_2', 'layer_3']:
        acc = metrics.get(f'{layer}_accuracy', 0)
        prec = metrics.get(f'{layer}_precision', 0)
        rec = metrics.get(f'{layer}_recall', 0)
        f1 = metrics.get(f'{layer}_f1', 0)
        
        if acc > 0 or prec > 0 or rec > 0 or f1 > 0:
            print(f"    {layer.upper()}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

def _handle_training_phase_progress(phase: str, current: int, total: int, message: str, 
                                   progress_bars: dict, verbose: bool, use_tqdm: bool, **kwargs) -> None:
    """Handle progress updates for training phases."""
    phase_display = _format_phase_display(phase)
    
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
        batch_idx = kwargs.get('batch_idx', 0)
        batch_total = kwargs.get('batch_total', 1)
        metrics = kwargs.get('metrics', {})
        phase_num = _get_phase_number(phase)
        
        if use_tqdm:
            _handle_tqdm_training_progress(phase, current, total, epoch, batch_idx, 
                                         batch_total, metrics, phase_num, progress_bars, verbose)
        else:
            _handle_simple_training_progress(phase, current, total, epoch, batch_idx,
                                           batch_total, metrics, phase_num, verbose)
    else:
        if use_tqdm:
            _handle_tqdm_phase_progress(phase, current, total, message, progress_bars, phase_display)
        else:
            percentage = (current / total) * 100 if total > 0 else 0
            print(f"🚀 {phase_display} ({percentage:.0f}%): {message}")

def _handle_tqdm_training_progress(phase: str, current: int, total: int, epoch: int, 
                                 batch_idx: int, batch_total: int, metrics: dict, 
                                 phase_num: str, progress_bars: dict, verbose: bool) -> None:
    """Handle tqdm progress for training phases."""
    # Create or update epoch progress bar
    epoch_bar_key = f"{phase}_epoch"
    if epoch_bar_key not in progress_bars:
        progress_bars[epoch_bar_key] = tqdm(
            total=total, desc=f"Phase {phase_num} Epochs", unit="epoch", position=0, leave=True
        )
    
    epoch_bar = progress_bars[epoch_bar_key]
    
    # Handle batch progress
    batch_bar_key = f"{phase}_batch_{epoch}"
    if batch_bar_key not in progress_bars and batch_total > 1:
        progress_bars[batch_bar_key] = tqdm(
            total=batch_total, desc=f"Phase {phase_num} Epoch {epoch} Batches", 
            unit="batch", position=1, leave=False
        )
    
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
    
    # Update epoch progress when complete
    if current == total:
        if batch_bar_key in progress_bars:
            progress_bars[batch_bar_key].close()
            del progress_bars[batch_bar_key]
        
        epoch_postfix = {}
        if metrics:
            loss = metrics.get('train_loss', 0)
            val_loss = metrics.get('val_loss', 0)
            avg_acc, avg_f1 = _calculate_layer_averages(metrics)
            
            epoch_postfix.update({
                'Train_Loss': f"{loss:.4f}",
                'Val_Loss': f"{val_loss:.4f}",
                'Avg_Acc': f"{avg_acc:.3f}",
                'Avg_F1': f"{avg_f1:.3f}"
            })
        
        epoch_bar.set_postfix(epoch_postfix)
        epoch_bar.update(1)
        
        if current >= total:
            epoch_bar.close()
            del progress_bars[epoch_bar_key]
            _print_layer_metrics(metrics, phase_num, epoch, verbose)

def _handle_simple_training_progress(phase: str, current: int, total: int, epoch: int,
                                   batch_idx: int, batch_total: int, metrics: dict,
                                   phase_num: str, verbose: bool) -> None:
    """Handle simple text progress for training phases."""
    if batch_idx > 0 and batch_total > 0 and verbose and metrics:
        batch_pct = (batch_idx / batch_total) * 100
        loss = metrics.get('train_loss', 0)
        layer1_acc = metrics.get('layer_1_accuracy', 0)
        layer1_f1 = metrics.get('layer_1_f1', 0)
        print(f"🔄 Phase {phase_num} - Epoch {epoch} ({batch_pct:.0f}%): "
              f"Loss: {loss:.4f} | L1_Acc: {layer1_acc:.3f} | L1_F1: {layer1_f1:.3f}")
    
    elif current == total and metrics:
        _print_layer_metrics(metrics, phase_num, epoch, verbose)

def _handle_tqdm_phase_progress(phase: str, current: int, total: int, message: str,
                               progress_bars: dict, phase_display: str) -> None:
    """Handle tqdm progress for non-training phases."""
    phase_bar_key = f"{phase}_phase"
    if phase_bar_key not in progress_bars:
        progress_bars[phase_bar_key] = tqdm(
            total=100, desc=phase_display, unit="%", position=0, leave=True
        )
    
    phase_bar = progress_bars[phase_bar_key]
    percentage = (current / total) * 100 if total > 0 else 0
    phase_bar.n = int(percentage)
    phase_bar.set_description(f"{phase_display}: {message}")
    phase_bar.refresh()
    
    if percentage >= 100:
        phase_bar.close()
        del progress_bars[phase_bar_key]

def create_progress_callback(use_tqdm: bool = True, verbose: bool = True):
    """Create a unified progress callback that can use tqdm or simple text output."""
    progress_bars = {} if use_tqdm else None
    
    def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        """Unified progress callback with configurable output mode."""
        phase_display = _format_phase_display(phase)
        
        if phase in ['training_phase_1', 'training_phase_2']:
            _handle_training_phase_progress(phase, current, total, message, 
                                          progress_bars, verbose, use_tqdm, **kwargs)
        else:
            percentage = (current / total) * 100 if total > 0 else 0
            
            if use_tqdm:
                _handle_tqdm_phase_progress(phase, current, total, message, progress_bars, phase_display)
            else:
                status_icon = "✅" if percentage == 100 else "🔄"
                print(f"{status_icon} {phase_display} ({percentage:.0f}%): {message}")
    
    return progress_callback

def _create_arg_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='SmartCash Unified Training Pipeline Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-phase training (traditional)
  %(prog)s --backbone cspdarknet --phase1-epochs 2 --phase2-epochs 3
  %(prog)s --backbone efficientnet_b4 --phase1-epochs 1 --phase2-epochs 1 --verbose
  
  # Single-phase training (flexible)
  %(prog)s --training-mode single_phase --phase1-epochs 3 --phase2-epochs 2  # 5 epochs unified training
  %(prog)s --training-mode single_phase --single-layer-mode single --phase1-epochs 2  # Single layer training
  %(prog)s --training-mode single_phase --single-freeze-backbone --phase1-epochs 3  # Frozen backbone
  %(prog)s --training-mode single_phase --single-layer-mode multi --phase1-epochs 5  # Multi-layer unfrozen
  
  # Other options
  %(prog)s --backbone cspdarknet --checkpoint-dir custom/checkpoints
  %(prog)s --loss-type focal --head-lr-p1 0.002 --head-lr-p2 0.0005 --backbone-lr 1e-6
  %(prog)s --batch-size 8 --loss-type weighted_multi_task --verbose
  %(prog)s --no-tqdm --verbose  # Use simple text progress instead of tqdm bars
  %(prog)s --no-early-stopping --phase1-epochs 5 --phase2-epochs 10  # Disable early stopping
  %(prog)s --patience 10 --es-metric val_loss --es-mode min --min-delta 0.01  # Custom early stopping
        """
    )
    
    # Model and training arguments
    parser.add_argument('--backbone', type=str, default='cspdarknet',
                       choices=['cspdarknet', 'efficientnet_b4'],
                       help='Model backbone architecture (default: cspdarknet)')
    parser.add_argument('--phase1-epochs', type=int, default=1,
                       help='Number of epochs for phase 1 (frozen backbone training) (default: 1)')
    parser.add_argument('--phase2-epochs', type=int, default=1,
                       help='Number of epochs for phase 2 (fine-tuning) (default: 1)')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                       help='Directory for checkpoint management (default: data/checkpoints)')
    parser.add_argument('--training-mode', type=str, default='two_phase',
                       choices=['single_phase', 'two_phase'],
                       help='Training mode: single_phase (unified training) or two_phase (freeze then fine-tune) (default: two_phase)')
    
    # Training configuration
    parser.add_argument('--loss-type', type=str, default='uncertainty_multi_task',
                       choices=['uncertainty_multi_task', 'weighted_multi_task', 'focal', 'standard'],
                       help='Loss function type (default: uncertainty_multi_task)')
    parser.add_argument('--head-lr-p1', type=float, default=1e-3,
                       help='Learning rate for detection heads in phase 1 (default: 1e-3)')
    parser.add_argument('--head-lr-p2', type=float, default=1e-4,
                       help='Learning rate for detection heads in phase 2 (default: 1e-4)')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                       help='Learning rate for backbone in phase 2 (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (default: auto-detected based on platform)')
    
    # Early stopping configuration
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping (default: enabled based on config)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping completely')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience - epochs to wait before stopping (default: 15)')
    parser.add_argument('--es-metric', type=str, default='val_map50',
                       choices=['val_map50', 'val_loss', 'train_loss', 'val_map75', 'val_accuracy'],
                       help='Metric to monitor for early stopping (default: val_map50)')
    parser.add_argument('--es-mode', type=str, default='max',
                       choices=['max', 'min'],
                       help='Early stopping mode - max for metrics that should increase, min for metrics that should decrease (default: max)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement for early stopping (default: 0.001)')
    
    # Single-phase specific options
    parser.add_argument('--single-layer-mode', type=str, default='multi',
                       choices=['single', 'multi'],
                       help='Layer mode for single-phase training: single (layer_1 only) or multi (all layers) (default: multi)')
    parser.add_argument('--single-freeze-backbone', action='store_true',
                       help='Freeze backbone during single-phase training (default: unfrozen)')
    
    # System and output arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose progress logging')
    parser.add_argument('--no-tqdm', action='store_true',
                       help='Disable tqdm progress bars (use simple text output instead)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU training mode (disables GPU/MPS autodetection)')
    
    return parser

def _print_configuration(args):
    """Print training configuration in a structured format."""
    print("=" * 80)
    print("🚀 SmartCash Unified Training Pipeline")
    print("=" * 80)
    
    # Basic configuration
    print(f"📋 Configuration:")
    print(f"   • Backbone: {args.backbone}")
    print(f"   • Training mode: {args.training_mode}")
    
    if args.training_mode == 'two_phase':
        print(f"   • Phase 1 epochs: {args.phase1_epochs} (frozen backbone)")
        print(f"   • Phase 2 epochs: {args.phase2_epochs} (fine-tuning)")
    else:
        # In single phase mode, only use phase1_epochs (ignore phase2_epochs)
        total_epochs = args.phase1_epochs
        print(f"   • Single phase epochs: {total_epochs} (unified training)")
        if args.phase2_epochs > 0:
            print(f"   • Note: phase2_epochs ({args.phase2_epochs}) ignored in single phase mode")
        print(f"   • Layer mode: {args.single_layer_mode}")
        backbone_status = "frozen" if args.single_freeze_backbone else "unfrozen"
        print(f"   • Backbone: {backbone_status}")
    
    print(f"   • Checkpoint directory: {args.checkpoint_dir}")
    
    # Training parameters
    print(f"\n🎛️ Training Parameters:")
    training_params = [
        ("Loss type", args.loss_type),
        ("Head LR (Phase 1)", args.head_lr_p1),
        ("Head LR (Phase 2)", args.head_lr_p2),
        ("Backbone LR", args.backbone_lr),
        ("Batch size", 'Auto-detected' if args.batch_size is None else args.batch_size),
        ("Verbose logging", 'Enabled' if args.verbose else 'Disabled'),
        ("Progress bars", 'Simple text' if args.no_tqdm else 'tqdm (visual)'),
        ("Device mode", 'CPU (forced)' if args.force_cpu else 'Auto-detect')
    ]
    
    for param_name, param_value in training_params:
        print(f"   • {param_name}: {param_value}")
    
    # Early stopping configuration
    print(f"\n🛑 Early Stopping Configuration:")
    if args.no_early_stopping:
        es_status = "Disabled"
    elif args.early_stopping:
        es_status = "Enabled (forced)"
    else:
        es_status = "Enabled (config default)"
    
    print(f"   • Status: {es_status}")
    if not args.no_early_stopping:
        print(f"   • Patience: {args.patience} epochs")
        print(f"   • Metric: {args.es_metric}")
        mode_desc = 'increase' if args.es_mode == 'max' else 'decrease'
        print(f"   • Mode: {args.es_mode} (better values should {mode_desc})")
        print(f"   • Min delta: {args.min_delta}")
    
    print("\n" + "=" * 80)

def _print_layer_metrics_detailed(best_metrics: dict) -> None:
    """Print detailed layer-wise performance metrics."""
    print(f"\n📊 Layer-wise Performance Metrics:")
    for layer in ['layer_1', 'layer_2', 'layer_3']:
        layer_display = layer.upper().replace('_', ' ')
        accuracy = best_metrics.get(f'{layer}_accuracy', 0)
        precision = best_metrics.get(f'{layer}_precision', 0)
        recall = best_metrics.get(f'{layer}_recall', 0)
        f1_score = best_metrics.get(f'{layer}_f1', 0)
        
        if accuracy > 0 or precision > 0 or recall > 0 or f1_score > 0:
            print(f"   • {layer_display}:")
            metric_lines = [
                f"     - Accuracy:  {accuracy:.4f}",
                f"     - Precision: {precision:.4f}",
                f"     - Recall:    {recall:.4f}",
                f"     - F1 Score:  {f1_score:.4f}"
            ]
            print('\n'.join(metric_lines))
        else:
            print(f"   • {layer_display}: No metrics available")

def _print_overall_performance(best_metrics: dict) -> None:
    """Print overall performance averages."""
    layer_accuracies = [best_metrics.get(f'{layer}_accuracy', 0) for layer in ['layer_1', 'layer_2', 'layer_3']]
    layer_f1_scores = [best_metrics.get(f'{layer}_f1', 0) for layer in ['layer_1', 'layer_2', 'layer_3']]
    
    if any(acc > 0 for acc in layer_accuracies):
        avg_accuracy = sum(layer_accuracies) / len([acc for acc in layer_accuracies if acc > 0])
        avg_f1 = sum(layer_f1_scores) / len([f1 for f1 in layer_f1_scores if f1 > 0])
        
        print(f"\n🎯 Overall Performance:")
        print(f"   • Average Accuracy: {avg_accuracy:.4f}")
        print(f"   • Average F1 Score: {avg_f1:.4f}")

def _print_visualization_results(viz_result: dict) -> None:
    """Print visualization results and chart information."""
    if not viz_result.get('success'):
        return
        
    charts_count = viz_result.get('charts_count', 0)
    session_id = viz_result.get('session_id', 'N/A')
    generated_charts = viz_result.get('generated_charts', {})
    
    print(f"\n📊 Visualization Results:")
    print(f"   • Charts generated: {charts_count}")
    print(f"   • Session ID: {session_id}")
    print(f"   • Charts location: data/visualization/{session_id}/")
    
    if generated_charts:
        print(f"\n📁 Generated Chart Files:")
        chart_categories = {
            'Training Analysis': ['training_curves', 'phase_analysis', 'lr_schedule'],
            'Performance Metrics': ['layer_metrics', 'dashboard'],
            'Confusion Matrices': [k for k in generated_charts.keys() if 'confusion_matrix' in k]
        }
        
        for category, chart_types in chart_categories.items():
            matching_charts = {k: v for k, v in generated_charts.items() 
                             if any(chart_type in k for chart_type in chart_types)}
            
            if matching_charts:
                print(f"   • {category}:")
                for chart_name, chart_path in matching_charts.items():
                    chart_display = chart_name.replace('_', ' ').title()
                    file_name = Path(chart_path).name
                    print(f"     - {chart_display}: {file_name}")
        
        print(f"\n🔗 Quick Access Commands:")
        print(f"   • View all charts: open data/visualization/{session_id}/")
        print(f"   • Dashboard: open data/visualization/{session_id}/comprehensive_dashboard.png")
        if 'training_curves' in generated_charts:
            print(f"   • Training curves: open data/visualization/{session_id}/training_curves.png")

def _print_training_summary(args, pipeline_summary: dict, viz_result: dict) -> None:
    """Print comprehensive training summary."""
    total_duration = pipeline_summary.get('total_duration', 0)
    generated_charts = viz_result.get('generated_charts', {})
    session_id = viz_result.get('session_id', 'N/A')
    
    print(f"\n📋 Complete Training Summary:")
    summary_items = [
        ("Backbone", args.backbone),
        ("Training mode", args.training_mode),
        ("Total phases", "6 (all completed)"),
        ("Training duration", f"{total_duration:.1f}s")
    ]
    
    if args.training_mode == 'two_phase':
        summary_items.extend([
            ("Phase 1 epochs", args.phase1_epochs),
            ("Phase 2 epochs", args.phase2_epochs)
        ])
    else:
        # In single phase mode, only use phase1_epochs (ignore phase2_epochs)
        total_epochs = args.phase1_epochs
        backbone_status = "frozen" if args.single_freeze_backbone else "unfrozen"
        summary_items.extend([
            ("Single phase epochs", total_epochs),
            ("Layer mode", args.single_layer_mode),
            ("Backbone", backbone_status)
        ])
        if args.phase2_epochs > 0:
            summary_items.append(("Note", f"phase2_epochs ({args.phase2_epochs}) ignored in single phase mode"))
    
    for item_name, item_value in summary_items:
        print(f"   • {item_name}: {item_value}")
    
    if viz_result.get('success') and generated_charts:
        print(f"   • Visualization session: {session_id}")
        print(f"   • Charts directory: data/visualization/{session_id}/")
        print(f"   • Key charts:")
        key_charts = ['dashboard', 'training_curves', 'layer_metrics']
        for chart_key in key_charts:
            if chart_key in generated_charts:
                chart_file = Path(generated_charts[chart_key]).name
                print(f"     - {chart_key.replace('_', ' ').title()}: {chart_file}")

def _process_training_results(result: dict, args) -> int:
    """Process and display training results."""
    if result.get('success'):
        print("\n" + "=" * 80)
        print("🎉 Unified Training Pipeline Completed Successfully!")
        print("=" * 80)
        
        # Display pipeline summary
        pipeline_summary = result.get('pipeline_summary', {})
        total_duration = pipeline_summary.get('total_duration', 0)
        phases_completed = pipeline_summary.get('phases_completed', 0)
        
        print(f"📊 Pipeline Summary:")
        print(f"   • Total duration: {total_duration:.1f} seconds")
        print(f"   • Phases completed: {phases_completed}/6")
        print(f"   • Overall success: {'✅' if pipeline_summary.get('success', False) else '❌'}")
        
        # Display training results
        training_result = result.get('final_training_result', {})
        if training_result.get('success'):
            best_metrics = training_result.get('best_metrics', {})
            best_checkpoint = training_result.get('best_checkpoint')
            
            print(f"\n🏆 Training Results:")
            print(f"   • Best mAP@0.5: {best_metrics.get('val_map50', 0):.4f}")
            print(f"   • Final train loss: {best_metrics.get('train_loss', 0):.4f}")
            print(f"   • Final val loss: {best_metrics.get('val_loss', 0):.4f}")
            
            _print_layer_metrics_detailed(best_metrics)
            
            print(f"\n💾 Model Checkpoint:")
            print(f"   • Best checkpoint: {Path(best_checkpoint).name if best_checkpoint else 'N/A'}")
            
            _print_overall_performance(best_metrics)
        
        # Display visualization results
        viz_result = result.get('visualization_result', {})
        _print_visualization_results(viz_result)
        
        print(f"\n💾 Checkpoint Management:")
        print(f"   • All checkpoints saved to: {args.checkpoint_dir}")
        print(f"   • Unified naming convention applied")
        
        _print_training_summary(args, pipeline_summary, viz_result)
        
        print(f"\n🎉 Training completed successfully! All results and visualizations saved.")
        return 0
        
    else:
        print("\n" + "=" * 80)
        print("❌ Unified Training Pipeline Failed")
        print("=" * 80)
        error_msg = result.get('error', 'Unknown error')
        print(f"Error: {error_msg}")
        
        # Show partial results if available
        pipeline_summary = result.get('pipeline_summary', {})
        if pipeline_summary:
            phases_completed = pipeline_summary.get('phases_completed', 0)
            print(f"Phases completed before failure: {phases_completed}/6")
        
        return 1

def main():
    """Main function demonstrating the unified training pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = _create_arg_parser()
    args = parser.parse_args()
    
    # Print configuration
    _print_configuration(args)
    
    try:
        # Create progress callback (tqdm or simple text-based)
        use_tqdm = not args.no_tqdm
        progress_callback = create_progress_callback(use_tqdm, args.verbose)
        print(f"📊 Using {'tqdm progress bars' if use_tqdm else 'simple text progress output'}")
        
        # Run the unified training pipeline
        print("🚀 Starting unified training pipeline...")
        print()
        
        result = run_full_training_pipeline(
            backbone=args.backbone,
            phase_1_epochs=args.phase1_epochs,
            phase_2_epochs=args.phase2_epochs,
            checkpoint_dir=args.checkpoint_dir,
            progress_callback=progress_callback,
            verbose=args.verbose,
            force_cpu=args.force_cpu,
            training_mode=args.training_mode,
            # Single-phase specific parameters
            single_phase_layer_mode=args.single_layer_mode,
            single_phase_freeze_backbone=args.single_freeze_backbone,
            # Training configuration parameters
            loss_type=args.loss_type,
            head_lr_p1=args.head_lr_p1,
            head_lr_p2=args.head_lr_p2,
            backbone_lr=args.backbone_lr,
            batch_size=args.batch_size,
            # Early stopping configuration parameters
            early_stopping_enabled=not args.no_early_stopping,
            early_stopping_patience=args.patience,
            early_stopping_metric=args.es_metric,
            early_stopping_mode=args.es_mode,
            early_stopping_min_delta=args.min_delta
        )
        
        # Process results
        return _process_training_results(result, args)
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())