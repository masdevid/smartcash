#!/usr/bin/env python3
"""
Training Arguments Helper

This module provides argument parsing utilities for SmartCash training examples.
It contains the common argument parser configuration used across different training examples.
"""

import argparse
from typing import Any


def create_training_arg_parser(description: str = "SmartCash Training Pipeline") -> argparse.ArgumentParser:
    """
    Create and configure argument parser for training examples.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-phase training (traditional)
  %(prog)s --backbone cspdarknet --phase1-epochs 2 --phase2-epochs 3
  %(prog)s --backbone efficientnet_b4 --phase1-epochs 1 --phase2-epochs 1 --verbose
  %(prog)s --backbone efficientnet_b4 --pretrained --phase1-epochs 2 --phase2-epochs 1  # With pretrained weights
  
  # Single-phase training (flexible)
  %(prog)s --training-mode single_phase --phase1-epochs 3 --phase2-epochs 2  # 5 epochs unified training
  %(prog)s --training-mode single_phase --single-layer-mode single --phase1-epochs 2  # Single layer training
  %(prog)s --training-mode single_phase --single-freeze-backbone --phase1-epochs 3  # Frozen backbone
  %(prog)s --training-mode single_phase --single-layer-mode multi --phase1-epochs 5  # Multi-layer unfrozen
  %(prog)s --training-mode single_phase --pretrained --phase1-epochs 3  # Single phase with pretrained weights
  
  # Optimizer and scheduler options
  %(prog)s --optimizer adamw --scheduler cosine --weight-decay 1e-2 --cosine-eta-min 1e-6
  %(prog)s --optimizer sgd --scheduler step --weight-decay 5e-4
  %(prog)s --optimizer adam --scheduler plateau --weight-decay 1e-3
  
  # Resume training options
  %(prog)s --resume  # Auto-detect latest last_*.pt checkpoint
  %(prog)s --resume data/checkpoints/best_model.pt  # Resume from specific checkpoint
  %(prog)s --resume --resume-optimizer --resume-scheduler  # Auto-detect with optimizer/scheduler state
  %(prog)s --resume data/checkpoints/epoch_10.pt --resume-epoch 5  # Resume from specific epoch
  
  # Phase jumping options
  %(prog)s --start-phase 2  # Jump directly to Phase 2 using backup best_phase1 model
  %(prog)s --start-phase 2 --resume  # Resume Phase 2 training from last checkpoint
  
  # Other options
  %(prog)s --backbone cspdarknet --checkpoint-dir custom/checkpoints
  %(prog)s --loss-type focal --head-lr-p1 0.002 --head-lr-p2 0.0005 --backbone-lr 1e-6
  %(prog)s --batch-size 8 --loss-type weighted_multi_task --verbose
  %(prog)s --no-tqdm --verbose  # Use simple text progress instead of tqdm bars
  %(prog)s --no-early-stopping --phase1-epochs 5 --phase2-epochs 10  # Disable early stopping
  %(prog)s --patience 20 --es-metric val_accuracy --es-mode max --min-delta 0.01  # Custom early stopping
  %(prog)s --phase-specific-early-stopping --phase1-epochs 5 --phase2-epochs 10  # Smart phase-aware early stopping
  %(prog)s --debug-map --verbose  # Enable hierarchical mAP debug logging
        """
    )
    
    # Model and training arguments
    parser.add_argument('--backbone', type=str, default='cspdarknet',
                       choices=['cspdarknet', 'efficientnet_b4'],
                       help='Model backbone architecture (default: cspdarknet)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights for backbone (default: False, train from scratch)')
    parser.add_argument('--use-smartcash-architecture', action='store_true',
                       help='Use new SmartCash YOLOv5 architecture (17 classes with post-inference mapping)')
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
    parser.add_argument('--head-lr-p1', type=float, default=0.001,
                       help='Learning rate for detection heads in phase 1 (default: 0.001)')
    parser.add_argument('--head-lr-p2', type=float, default=0.0001,
                       help='Learning rate for detection heads in phase 2 (default: 0.0001)')
    parser.add_argument('--backbone-lr', type=float, default=1e-05,
                       help='Learning rate for backbone (default: 1e-05)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16, multiply LR by 3x when set to 32)')
    
    # Optimizer configuration
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd', 'rmsprop'],
                       help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'exponential', 'linear'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help='Weight decay for optimizer (default: 1e-2)')
    parser.add_argument('--cosine-eta-min', type=float, default=1e-6,
                       help='Minimum learning rate for cosine scheduler (default: 1e-6)')
    
    # Resume training configuration
    parser.add_argument('--resume', nargs='?', const='auto', default=None,
                       help='Resume training from checkpoint. Use --resume to auto-detect latest last_*.pt, or --resume PATH to specify checkpoint file')
    parser.add_argument('--resume-optimizer', action='store_true',
                       help='Resume optimizer state when resuming training (default: False)')
    parser.add_argument('--resume-scheduler', action='store_true',
                       help='Resume scheduler state when resuming training (default: False)')
    parser.add_argument('--resume-epoch', type=int, default=None,
                       help='Specific epoch to resume from (overrides checkpoint epoch)')
    
    # Phase jumping configuration
    parser.add_argument('--start-phase', type=int, default=1,
                       choices=[1, 2],
                       help='Starting phase for training: 1 (start from Phase 1) or 2 (jump directly to Phase 2) (default: 1)')
    
    # Early stopping configuration
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping (default: enabled based on config)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping completely')
    parser.add_argument('--phase-specific-early-stopping', action='store_true',
                       help='Enable phase-specific early stopping with custom criteria for Phase 1 and Phase 2')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience - epochs to wait before stopping (default: 15)')
    parser.add_argument('--es-metric', type=str, default='val_accuracy',
                       choices=['val_accuracy', 'val_loss', 'train_loss', 'val_f1', 'val_precision', 'val_recall'],
                       help='Metric to monitor for early stopping (default: val_accuracy)')
    parser.add_argument('--es-mode', type=str, default='max',
                       choices=['max', 'min'],
                       help='Early stopping mode - max for metrics that should increase, min for metrics that should decrease (default: max)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement for early stopping (default: 0.001)')
    
    # Note: Validation metrics now always use both YOLOv5 hierarchical and per-layer metrics
    
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
    parser.add_argument('--debug-map', action='store_true',
                       help='Enable debug logging for hierarchical mAP calculations (default: False)')
    
    return parser


def print_training_configuration(args: Any) -> None:
    """
    Print training configuration in a structured format.
    
    Args:
        args: Parsed arguments from ArgumentParser
    """
    print("=" * 80)
    print("🚀 SmartCash Training Pipeline")
    print("=" * 80)
    
    # Basic configuration
    print(f"📋 Configuration:")
    print(f"   • Backbone: {args.backbone}")
    print(f"   • Pretrained weights: {'Enabled' if args.pretrained else 'Disabled (training from scratch)'}")
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
    
    # Show checkpoint naming example
    example_layer_mode = 'single' if args.training_mode == 'single_phase' and args.single_layer_mode == 'single' else 'multi'
    example_freeze_status = 'frozen' if args.training_mode == 'single_phase' and args.single_freeze_backbone else 'unfrozen'
    example_pretrained = '_pretrained' if args.pretrained else ''
    example_checkpoint_name = f"best_{args.backbone}_{args.training_mode}_{example_layer_mode}_{example_freeze_status}{example_pretrained}_YYYYMMDD.pt"
    print(f"   • Checkpoint naming: {example_checkpoint_name}")
    
    # Learning Rate Configuration
    print(f"\n📊 Learning Rate Configuration:")
    
    # Format learning rates - use scientific notation for default values
    head_lr_p1_display = "1e-03" if args.head_lr_p1 == 1e-3 else f"{args.head_lr_p1}"
    head_lr_p2_display = "1e-04" if args.head_lr_p2 == 1e-4 else f"{args.head_lr_p2}"
    backbone_lr_display = "1e-05" if args.backbone_lr == 1e-5 else f"{args.backbone_lr}"
    
    print(f"   • Head LR (Phase 1): {head_lr_p1_display}")
    print(f"   • Head LR (Phase 2): {head_lr_p2_display}")
    print(f"   • Backbone LR: {backbone_lr_display}")
    
    # Training parameters
    print(f"\n🎛️ Training Parameters:")
    training_params = [
        ("Loss type", args.loss_type),
        ("Batch size", 'Auto-detected' if args.batch_size is None else args.batch_size),
        ("Verbose logging", 'Enabled' if args.verbose else 'Disabled'),
        ("Progress bars", 'Simple text' if args.no_tqdm else 'tqdm (visual)'),
        ("Device mode", 'CPU (forced)' if args.force_cpu else 'Auto-detect'),
        ("Debug mAP", 'Enabled' if args.debug_map else 'Disabled')
    ]
    
    for param_name, param_value in training_params:
        print(f"   • {param_name}: {param_value}")
    
    # Optimizer and scheduler parameters
    print(f"\n⚙️ Optimizer & Scheduler Configuration:")
    optimizer_params = [
        ("Optimizer", args.optimizer),
        ("Scheduler", args.scheduler),
        ("Weight decay", args.weight_decay),
        ("Cosine eta min", args.cosine_eta_min if args.scheduler == 'cosine' else 'N/A (not using cosine scheduler)')
    ]
    
    for param_name, param_value in optimizer_params:
        print(f"   • {param_name}: {param_value}")
    
    # Resume training parameters
    print(f"\n🔄 Resume Training Configuration:")
    if args.resume:
        resume_params = [
            ("Resume checkpoint", args.resume),
            ("Resume optimizer state", 'Yes' if args.resume_optimizer else 'No'),
            ("Resume scheduler state", 'Yes' if args.resume_scheduler else 'No'),
            ("Resume epoch override", args.resume_epoch if args.resume_epoch is not None else 'Use checkpoint epoch')
        ]
        
        for param_name, param_value in resume_params:
            print(f"   • {param_name}: {param_value}")
    else:
        print("   • Resume training: Disabled (training from scratch)")
    
    # Phase jumping configuration
    print(f"\n🚀 Phase Configuration:")
    if args.start_phase == 2:
        print(f"   • Start phase: {args.start_phase} (jumping directly to Phase 2)")
        if args.resume:
            print(f"   • Phase 2 mode: Resume from standard best model (best_{{backbone}}_{{date}}.pt)")
        else:
            print(f"   • Phase 2 mode: Override standard best model with backup_phase1, then load standard best model")
    else:
        print(f"   • Start phase: {args.start_phase} (normal two-phase training)")
    
    # Early stopping configuration
    print(f"\n🛑 Early Stopping Configuration:")
    if args.no_early_stopping:
        es_status = "Disabled"
    elif args.phase_specific_early_stopping:
        es_status = "Phase-specific (smart criteria)"
    elif args.early_stopping:
        es_status = "Enabled (forced)"
    else:
        es_status = "Enabled (config default)"
    
    print(f"   • Status: {es_status}")
    if not args.no_early_stopping:
        if args.phase_specific_early_stopping:
            print(f"   • Type: Phase-specific with custom criteria")
            print(f"   • Phase 1: Train loss plateau + val_accuracy stability")
            print(f"   • Phase 2: F1/mAP improvement + overfitting detection")
        else:
            print(f"   • Patience: {args.patience} epochs")
            print(f"   • Metric: {args.es_metric}")
            mode_desc = 'increase' if args.es_mode == 'max' else 'decrease'
            print(f"   • Mode: {args.es_mode} (better values should {mode_desc})")
            print(f"   • Min delta: {args.min_delta}")
    
    print("\n" + "=" * 80)


def get_training_kwargs(args: Any) -> dict:
    """
    Convert parsed arguments to keyword arguments for run_full_training_pipeline.
    
    Args:
        args: Parsed arguments from ArgumentParser
        
    Returns:
        Dictionary of keyword arguments for the training pipeline
    """
    return {
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'use_smartcash_architecture': args.use_smartcash_architecture,
        'phase_1_epochs': args.phase1_epochs,
        'phase_2_epochs': args.phase2_epochs,
        'checkpoint_dir': args.checkpoint_dir,
        'force_cpu': args.force_cpu,
        'training_mode': args.training_mode,
        # Single-phase specific parameters
        'single_phase_layer_mode': args.single_layer_mode,
        'single_phase_freeze_backbone': args.single_freeze_backbone,
        # Training configuration parameters
        'loss_type': args.loss_type,
        'head_lr_p1': args.head_lr_p1,
        'head_lr_p2': args.head_lr_p2,
        'backbone_lr': args.backbone_lr,
        'batch_size': args.batch_size,
        # Optimizer and scheduler configuration parameters
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'weight_decay': args.weight_decay,
        'cosine_eta_min': args.cosine_eta_min,
        # Resume training configuration parameters
        'resume_checkpoint': args.resume,
        'resume_optimizer_state': args.resume_optimizer,
        'resume_scheduler_state': args.resume_scheduler,
        'resume_epoch': args.resume_epoch,
        # Phase jumping configuration parameters
        'start_phase': args.start_phase,
        # Early stopping configuration parameters
        'early_stopping_enabled': not args.no_early_stopping,
        'early_stopping_patience': args.patience,
        'patience': args.patience,  # Also include direct patience mapping for fallback compatibility
        'early_stopping_metric': args.es_metric,
        'early_stopping_mode': args.es_mode,
        'early_stopping_min_delta': args.min_delta,
        'phase_specific_early_stopping': args.phase_specific_early_stopping,
        # Validation metrics: Always use both YOLOv5 hierarchical and per-layer metrics
        'validation_metrics_config': {
            'use_hierarchical_validation': True
        },
        # Debug configuration
        'debug_map': args.debug_map
    }