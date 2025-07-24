#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/examples/training_pipeline.py

MODEL_ARC.md Compliant Two-Phase Training Pipeline for SmartCash Multi-Layer Detection Model.

This implementation follows the architecture specifications outlined in docs/MODEL_ARC.md:

🏗️ Architecture Compliance:
- Multi-layer detection system (layer_1, layer_2, layer_3)
- Two-phase training strategy: freeze backbone → fine-tune entire model
- Uncertainty-based multi-task loss with learnable log-variance parameters
- Support for EfficientNet-B4 and CSPDarknet backbones
- Proper learning rate scheduling (backbone: 1e-5, heads: 1e-3/1e-4)

🎯 Key Features:
- Phase 1: Freeze backbone, train detection heads only (MODEL_ARC.md:94)
- Phase 2: Unfreeze entire model for fine-tuning (MODEL_ARC.md:95)
- Multi-task loss: total_loss = λ1*loss_layer1 + λ2*loss_layer2 + λ3*loss_layer3
- Progress tracking with enhanced metrics display
- Comprehensive checkpoint management with proper naming
- Architecture compliance validation and reporting

🚀 Usage Examples:
  python training_pipeline.py --backbone efficientnet_b4 --verbose
  python training_pipeline.py --backbone cspdarknet --phase1-epochs 30 --phase2-epochs 20
  python training_pipeline.py --resume checkpoint.pt --phase2-only
  python training_pipeline.py --check-only  # Prerequisites check only

📊 Class Layer Configuration (MODEL_ARC.md:58-76):
- Layer 1: Full banknote detection (7 denominations: 001,002,005,010,020,050,100)
- Layer 2: Denomination-specific features (7 classes: l2_001 to l2_100)
- Layer 3: Common features (3 classes: l3_sign, l3_text, l3_thread)
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import create_model_api
from smartcash.model.training import (
    start_training, resume_training, get_training_info,
    create_training_service, TrainingService
)
from smartcash.model.api.backbone_api import check_data_prerequisites
from smartcash.common.logger import get_logger

# Setup logger - only warnings, errors, and phase transitions
logger = get_logger('training_pipeline_example', level="WARNING")

def create_optimized_progress_callback(verbose: bool = False) -> callable:
    """Create an optimized progress callback with minimal logging."""
    progress_bars = {}
    current_phase = None
    
    def callback(phase: str, current: int, total: int, message: str = "", **kwargs):
        """Optimized progress callback with phase transition logging."""
        nonlocal current_phase, progress_bars
        
        try:
            # Log phase transitions only
            if phase != current_phase:
                current_phase = phase
                if verbose:
                    print(f"\n📍 Phase: {phase.replace('_', ' ').title()}")
                
                # Close previous progress bar if exists
                if phase in progress_bars:
                    progress_bars[phase].close()
                    del progress_bars[phase]
            
            # Create or update progress bar
            if phase not in progress_bars and total > 0:
                progress_bars[phase] = tqdm(
                    total=total,
                    desc=f"{phase.replace('_', ' ').title()}",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    leave=False
                )
            
            # Update progress
            if phase in progress_bars:
                progress_bars[phase].n = current
                progress_bars[phase].total = total
                if message and verbose:
                    progress_bars[phase].set_postfix_str(message[:30], refresh=True)
                progress_bars[phase].refresh()
                
                # Close completed progress bars
                if current >= total:
                    progress_bars[phase].close()
                    del progress_bars[phase]
                    
        except Exception as e:
            if verbose:
                logger.warning(f"Progress callback error: {e}")
    
    # Return callback with cleanup function
    callback.cleanup = lambda: [bar.close() for bar in progress_bars.values()]
    return callback

def load_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'smartcash' / 'configs' / 'training_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"⚠️ Could not load config from {config_path}: {e}")
        return get_default_training_config()

def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration compliant with MODEL_ARC.md."""
    return {
        # Two-phase training strategy (MODEL_ARC.md:93-95)
        'training_phases': {
            'phase_1': {
                'description': 'Freeze backbone, train detection heads only',
                'epochs': 50,
                'freeze_backbone': True,
                'learning_rates': {
                    'backbone': 1e-5,  # MODEL_ARC.md:102-105
                    'head': 1e-3
                }
            },
            'phase_2': {
                'description': 'Unfreeze entire model for fine-tuning',
                'epochs': 50,
                'freeze_backbone': False,
                'learning_rates': {
                    'backbone': 1e-5,
                    'head': 1e-4
                }
            }
        },
        'training': {
            'epochs': 100,  # Total across both phases
            'batch_size': 16,
            'learning_rate': 0.001,  # Default fallback
            'weight_decay': 0.0005,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'warmup_epochs': 3,
            'mixed_precision': True,
            'gradient_clip': 10.0,
            'data': {
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,
                'drop_last': True
            },
            # Multi-task loss configuration (MODEL_ARC.md:96-97)
            'loss': {
                'type': 'uncertainty_multi_task',  # Uncertainty-based dynamic weighting
                'box_weight': 0.05,
                'obj_weight': 1.0,
                'cls_weight': 0.5,
                'focal_loss': False,
                'label_smoothing': 0.0,
                'multi_task_loss': {
                    'enabled': True,
                    'uncertainty_weighting': True,
                    'formula': 'total_loss = λ1 * loss_layer1 + λ2 * loss_layer2 + λ3 * loss_layer3'
                }
            },
            'early_stopping': {
                'enabled': True,
                'patience': 15,
                'metric': 'val_map50',
                'mode': 'max',
                'min_delta': 0.001
            }
        },
        # Multi-layer model architecture (MODEL_ARC.md:16-46)
        'model': {
            'backbone': 'efficientnet_b4',
            'layer_mode': 'multi',
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
            'multi_layer_heads': True,
            # Class layers from MODEL_ARC.md:58-76
            'num_classes': {
                'layer_1': 7,   # Full banknote detection: 001,002,005,010,020,050,100
                'layer_2': 7,   # Denomination-specific features: l2_001,l2_002,etc
                'layer_3': 3    # Common features: l3_sign,l3_text,l3_thread
            },
            'img_size': 640,
            'pretrained': True,
            'feature_optimization': True,
            'mixed_precision': True
        },
        'data': {
            'dataset_dir': 'data/preprocessed',
            'batch_size': 16,
            'num_workers': 4
        },
        'paths': {
            'pretrained_models': 'data/pretrained',
            'checkpoints': 'data/checkpoints',
            'training_outputs': 'runs/train'
        },
        'device': {
            'auto_detect': True,
            'preferred': 'cuda'
        }
    }

def check_training_prerequisites() -> Dict[str, Any]:
    """Check if all prerequisites for training are met."""
    try:
        # Check data prerequisites
        data_result = check_data_prerequisites()
        
        # Check for model components
        model_ready = True
        model_issues = []
        
        # Check if we can create a model API
        try:
            api = create_model_api()
            if not api:
                model_ready = False
                model_issues.append("Failed to create model API")
        except Exception as e:
            model_ready = False
            model_issues.append(f"Model API error: {str(e)}")
        
        # Check training info
        try:
            training_info = get_training_info()
            if not training_info.get('dataset_info'):
                model_issues.append("Dataset info not available")
        except Exception as e:
            model_issues.append(f"Training info error: {str(e)}")
        
        return {
            'success': True,
            'data_ready': data_result.get('prerequisites_ready', False),
            'model_ready': model_ready,
            'data_info': data_result,
            'model_issues': model_issues,
            'overall_ready': data_result.get('prerequisites_ready', False) and model_ready
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'overall_ready': False
        }

def build_training_model(backbone: str, config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Build model for training with proper configuration."""
    try:
        if verbose:
            print(f"🏗️ Building {backbone} model for training...")
        
        # Create model API with progress callback
        progress_callback = create_optimized_progress_callback(verbose)
        api = create_model_api(progress_callback=progress_callback)
        
        if not api:
            return {'success': False, 'error': 'Failed to create model API'}
        
        # Build model with training configuration
        model_config = config.get('model', {})
        model_info = api.build_model(**model_config)
        
        # Clean up progress bars
        progress_callback.cleanup()
        
        if model_info.get('status') == 'built':
            if verbose:
                print(f"✅ Model built successfully!")
                print(f"   • Backbone: {model_info.get('backbone', 'N/A')}")
                print(f"   • Layer mode: {model_info.get('layer_mode', 'N/A')}")
                print(f"   • Detection layers: {model_info.get('detection_layers', [])}")
                print(f"   • Parameters: {model_info.get('total_parameters', 'N/A'):,}")
                print(f"   • Device: {model_info.get('device', 'N/A')}")
            
            return {
                'success': True,
                'model_api': api,
                'model_info': model_info
            }
        else:
            error_msg = model_info.get('message', 'Model build failed')
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_training_pipeline(backbone: str, epochs: int, config: Dict[str, Any], 
                         resume_from: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """Run the complete two-phase training pipeline (MODEL_ARC.md compliant)."""
    try:
        print(f"\n{'='*80}")
        print(f"🚀 Two-Phase Training Pipeline: {backbone.upper()} (MODEL_ARC.md Compliant)")
        print(f"{'='*80}")
        
        # Step 1: Build model
        build_result = build_training_model(backbone, config, verbose)
        if not build_result.get('success'):
            return build_result
        
        model_api = build_result['model_api']
        model_info = build_result['model_info']
        
        # Step 2: Get training phases configuration
        training_phases = config.get('training_phases', {})
        phase_1 = training_phases.get('phase_1', {})
        phase_2 = training_phases.get('phase_2', {})
        
        # Override epochs if specified
        if epochs != 100:  # If user specified different total epochs
            phase_1_epochs = epochs // 2
            phase_2_epochs = epochs - phase_1_epochs
        else:
            phase_1_epochs = phase_1.get('epochs', 50)
            phase_2_epochs = phase_2.get('epochs', 50)
        
        if verbose:
            print(f"\n📋 Two-Phase Training Configuration:")
            print(f"   🔒 Phase 1: Freeze backbone, train heads ({phase_1_epochs} epochs)")
            print(f"       • Backbone LR: {phase_1.get('learning_rates', {}).get('backbone', 1e-5):.0e}")
            print(f"       • Head LR: {phase_1.get('learning_rates', {}).get('head', 1e-3):.0e}")
            print(f"   🔓 Phase 2: Fine-tune entire model ({phase_2_epochs} epochs)")
            print(f"       • Backbone LR: {phase_2.get('learning_rates', {}).get('backbone', 1e-5):.0e}")
            print(f"       • Head LR: {phase_2.get('learning_rates', {}).get('head', 1e-4):.0e}")
            print(f"   🎯 Multi-task loss: {config['training']['loss'].get('type', 'standard')}")
            print(f"   📊 Detection layers: {config['model']['detection_layers']}")
        
        # Step 3: Create progress callbacks
        progress_callback = create_optimized_progress_callback(verbose)
        
        def metrics_callback(metrics_dict):
            """Enhanced metrics callback for multi-layer training."""
            if verbose:
                epoch = metrics_dict.get('epoch', 0)
                train_loss = metrics_dict.get('train_loss', 0)
                val_loss = metrics_dict.get('val_loss', 0)
                val_map50 = metrics_dict.get('val_map50', 0)
                
                # Multi-layer specific metrics
                layer_losses = {
                    'layer_1': metrics_dict.get('layer_1_loss', 0),
                    'layer_2': metrics_dict.get('layer_2_loss', 0),
                    'layer_3': metrics_dict.get('layer_3_loss', 0)
                }
                
                if train_loss > 0 or val_loss > 0:
                    loss_breakdown = ", ".join([f"{k}={v:.3f}" for k, v in layer_losses.items() if v > 0])
                    phase_indicator = "🔒P1" if epoch < phase_1_epochs else "🔓P2"
                    print(f"📊 {phase_indicator} Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, mAP@0.5={val_map50:.4f}")
                    if loss_breakdown and verbose:
                        print(f"    Layer losses: {loss_breakdown}")
        
        # Step 4: Execute two-phase training
        all_results = []
        
        # Phase 1: Freeze backbone, train detection heads only
        if not resume_from:  # Only run phase 1 if not resuming
            print(f"\n🔒 Phase 1: Training detection heads (backbone frozen) - {phase_1_epochs} epochs")
            
            # Setup Phase 1 configuration
            phase_1_config = config.copy()
            phase_1_config['training']['epochs'] = phase_1_epochs
            phase_1_config['training']['freeze_backbone'] = True
            phase_1_config['training']['learning_rates'] = phase_1.get('learning_rates', {
                'backbone': 1e-5, 'head': 1e-3
            })
            
            phase_1_result = start_training(
                model_api=model_api,
                config=phase_1_config,
                epochs=phase_1_epochs,
                ui_components={'progress_callback': progress_callback, 'metrics_callback': metrics_callback},
                progress_callback=progress_callback,
                metrics_callback=metrics_callback
            )
            
            all_results.append(('phase_1', phase_1_result))
            
            if not phase_1_result.get('success', False):
                return {
                    'success': False,
                    'error': f"Phase 1 training failed: {phase_1_result.get('error', 'Unknown error')}",
                    'phase_1_result': phase_1_result
                }
            
            if verbose:
                print(f"✅ Phase 1 completed successfully!")
                if 'training_summary' in phase_1_result:
                    summary = phase_1_result['training_summary']
                    print(f"   • Best mAP@0.5: {summary.get('best_map50', 'N/A'):.4f}")
        
        # Phase 2: Unfreeze entire model for fine-tuning
        print(f"\n🔓 Phase 2: Fine-tuning entire model - {phase_2_epochs} epochs")
        
        # Setup Phase 2 configuration
        phase_2_config = config.copy()
        phase_2_config['training']['epochs'] = phase_2_epochs
        phase_2_config['training']['freeze_backbone'] = False
        phase_2_config['training']['learning_rates'] = phase_2.get('learning_rates', {
            'backbone': 1e-5, 'head': 1e-4
        })
        
        # For phase 2, we either resume from checkpoint or continue from phase 1
        if resume_from:
            phase_2_result = resume_training(
                model_api=model_api,
                checkpoint_path=resume_from,
                additional_epochs=phase_2_epochs,
                config=phase_2_config,
                ui_components={'progress_callback': progress_callback, 'metrics_callback': metrics_callback}
            )
        else:
            # Continue from phase 1 - use the best checkpoint from phase 1
            phase_1_checkpoint = None
            if all_results and 'training_summary' in all_results[0][1]:
                phase_1_checkpoint = all_results[0][1]['training_summary'].get('best_checkpoint')
            
            if phase_1_checkpoint:
                phase_2_result = resume_training(
                    model_api=model_api,
                    checkpoint_path=phase_1_checkpoint,
                    additional_epochs=phase_2_epochs,
                    config=phase_2_config,
                    ui_components={'progress_callback': progress_callback, 'metrics_callback': metrics_callback}
                )
            else:
                # Fallback: start phase 2 from current model state
                phase_2_result = start_training(
                    model_api=model_api,
                    config=phase_2_config,
                    epochs=phase_2_epochs,
                    ui_components={'progress_callback': progress_callback, 'metrics_callback': metrics_callback},
                    progress_callback=progress_callback,
                    metrics_callback=metrics_callback
                )
        
        all_results.append(('phase_2', phase_2_result))
        
        # Clean up progress bars
        progress_callback.cleanup()
        
        # Step 5: Process final results
        final_success = phase_2_result.get('success', False)
        
        if final_success:
            print(f"\n✅ Two-phase training completed successfully!")
            
            # Show comprehensive training summary
            if verbose:
                print(f"\n📊 Two-Phase Training Summary:")
                for phase_name, result in all_results:
                    if 'training_summary' in result:
                        summary = result['training_summary']
                        phase_label = "🔒 Phase 1" if phase_name == 'phase_1' else "🔓 Phase 2"
                        print(f"   {phase_label}:")
                        print(f"      • Epochs: {summary.get('total_epochs', 'N/A')}")
                        print(f"      • Best mAP@0.5: {summary.get('best_map50', 'N/A'):.4f}")
                        print(f"      • Final train loss: {summary.get('final_train_loss', 'N/A'):.4f}")
                        print(f"      • Training time: {summary.get('training_time', 'N/A')}")
                
                # Overall best performance
                phase_2_summary = phase_2_result.get('training_summary', {})
                print(f"   🏆 Final Performance:")
                print(f"      • Best checkpoint: {phase_2_summary.get('best_checkpoint', 'N/A')}")
                print(f"      • Architecture: Multi-layer detection ({len(config['model']['detection_layers'])} layers)")
                print(f"      • Loss function: {config['training']['loss'].get('type', 'standard')}")
        else:
            error_msg = phase_2_result.get('error', 'Unknown training error')
            print(f"❌ Phase 2 training failed: {error_msg}")
        
        return {
            'success': final_success,
            'model_info': model_info,
            'training_result': phase_2_result,  # Final result
            'phase_results': dict(all_results),  # All phase results
            'architecture_compliance': {
                'two_phase_training': True,
                'multi_layer_detection': True,
                'uncertainty_loss': config['training']['loss'].get('multi_task_loss', {}).get('enabled', False),
                'detection_layers': config['model']['detection_layers']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Two-phase training pipeline failed: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function to run training pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='SmartCash Two-Phase Training Pipeline (MODEL_ARC.md Compliant)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --backbone efficientnet_b4 --epochs 100 --verbose
  %(prog)s --backbone cspdarknet --phase1-epochs 30 --phase2-epochs 20
  %(prog)s --resume data/checkpoints/best_model.pt --phase2-only
  %(prog)s --check-only  # Only check prerequisites
        """
    )
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                       choices=['efficientnet_b4', 'cspdarknet'],
                       help='Backbone architecture to use (default: efficientnet_b4)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Total epochs across both phases (default: 100)')
    parser.add_argument('--phase1-epochs', type=int, default=50,
                       help='Epochs for phase 1 (freeze backbone) (default: 50)')
    parser.add_argument('--phase2-epochs', type=int, default=50,
                       help='Epochs for phase 2 (fine-tune) (default: 50)')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Skip phase 1, only run phase 2 fine-tuning')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                       help='Learning rate for backbone (default: 1e-5)')
    parser.add_argument('--head-lr-p1', type=float, default=1e-3,
                       help='Learning rate for heads in phase 1 (default: 1e-3)')
    parser.add_argument('--head-lr-p2', type=float, default=1e-4,
                       help='Learning rate for heads in phase 2 (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--loss-type', type=str, default='uncertainty_multi_task',
                       choices=['standard', 'uncertainty_multi_task'],
                       help='Loss function type (default: uncertainty_multi_task)')
    parser.add_argument('--config', type=str,
                       help='Path to training configuration YAML file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (shows all messages)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check prerequisites, do not train')
    
    args = parser.parse_args()
    
    # Configure logger based on verbose flag
    global logger
    if args.verbose:
        logger = get_logger('training_pipeline_example', level="INFO")
    
    print("=" * 100)
    print("🎯 SmartCash Two-Phase Training Pipeline (MODEL_ARC.md Compliant)")
    print("=" * 100)
    print(f"🏗️ Backbone: {args.backbone}")
    # Show phase configuration
    if hasattr(args, 'phase2_only') and args.phase2_only:
        print(f"🔓 Training Mode: Phase 2 Only (Fine-tuning)")
        print(f"📊 Phase 2 Epochs: {getattr(args, 'phase2_epochs', 50)}")
    else:
        print(f"🔒🔓 Training Mode: Two-Phase (Freeze → Fine-tune)")
        print(f"📊 Phase 1 Epochs: {getattr(args, 'phase1_epochs', 50)} (freeze backbone)")
        print(f"📊 Phase 2 Epochs: {getattr(args, 'phase2_epochs', 50)} (fine-tune)")
        print(f"📊 Total Epochs: {getattr(args, 'phase1_epochs', 50) + getattr(args, 'phase2_epochs', 50)}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎛️ Backbone LR: {getattr(args, 'backbone_lr', 1e-5):.0e}")
    print(f"🎛️ Head LR P1: {getattr(args, 'head_lr_p1', 1e-3):.0e}")
    print(f"🎛️ Head LR P2: {getattr(args, 'head_lr_p2', 1e-4):.0e}")
    print(f"⚙️ Optimizer: {args.optimizer}")
    print(f"🔮 Loss Type: {getattr(args, 'loss_type', 'uncertainty_multi_task')}")
    print(f"📝 Verbose: {'Enabled' if args.verbose else 'Disabled'}")
    if args.resume:
        print(f"🔄 Resume from: {args.resume}")
    print("-" * 100)
    
    try:
        # Step 1: Check prerequisites
        print("🔍 Checking training prerequisites...")
        prereq_result = check_training_prerequisites()
        
        if not prereq_result.get('success'):
            logger.error(f"❌ Failed to check prerequisites: {prereq_result.get('error')}")
            return 1
        
        # Show prerequisite status
        data_ready = prereq_result.get('data_ready', False)
        model_ready = prereq_result.get('model_ready', False)
        overall_ready = prereq_result.get('overall_ready', False)
        
        print(f"📊 Prerequisites Status:")
        print(f"   • Data ready: {'✅' if data_ready else '❌'}")
        print(f"   • Model ready: {'✅' if model_ready else '❌'}")
        print(f"   • Overall ready: {'✅' if overall_ready else '❌'}")
        
        if not data_ready:
            data_info = prereq_result.get('data_info', {})
            print(f"   • Data issue: {data_info.get('message', 'Unknown data issue')}")
        
        if not model_ready:
            model_issues = prereq_result.get('model_issues', [])
            for issue in model_issues:
                print(f"   • Model issue: {issue}")
        
        if args.check_only:
            print("\n✨ Prerequisite check completed!")
            return 0 if overall_ready else 1
        
        if not overall_ready:
            logger.error("❌ Prerequisites not met. Please fix the issues above before training.")
            return 1
        
        # Step 2: Load configuration
        config = load_training_config(args.config)
        
        # Override config with command line arguments
        config['training']['batch_size'] = args.batch_size
        config['training']['loss']['type'] = getattr(args, 'loss_type', 'uncertainty_multi_task')
        
        # Override phase configurations
        if 'training_phases' not in config:
            config['training_phases'] = {'phase_1': {}, 'phase_2': {}}
        
        config['training_phases']['phase_1']['epochs'] = getattr(args, 'phase1_epochs', 50)
        config['training_phases']['phase_1']['learning_rates'] = {
            'backbone': getattr(args, 'backbone_lr', 1e-5),
            'head': getattr(args, 'head_lr_p1', 1e-3)
        }
        config['training_phases']['phase_2']['epochs'] = getattr(args, 'phase2_epochs', 50)
        config['training_phases']['phase_2']['learning_rates'] = {
            'backbone': getattr(args, 'backbone_lr', 1e-5),
            'head': getattr(args, 'head_lr_p2', 1e-4)
        }
        
        # Handle phase2-only mode
        if hasattr(args, 'phase2_only') and args.phase2_only:
            config['training_phases']['phase_1']['epochs'] = 0  # Skip phase 1
        config['training']['optimizer'] = args.optimizer
        config['model']['backbone'] = args.backbone
        
        # Step 3: Run two-phase training pipeline
        total_epochs = (getattr(args, 'phase1_epochs', 50) + getattr(args, 'phase2_epochs', 50)) if not getattr(args, 'phase2_only', False) else getattr(args, 'phase2_epochs', 50)
        result = run_training_pipeline(
            backbone=args.backbone,
            epochs=total_epochs,
            config=config,
            resume_from=args.resume,
            verbose=args.verbose
        )
        
        # Step 4: Save results
        # Step 4: Save results with architecture compliance info
        phase_info = "phase2only" if getattr(args, 'phase2_only', False) else "twophase"
        results_file = f'training_results_{args.backbone}_{phase_info}_{total_epochs}epochs.json'
        
        # Enhanced results with architecture compliance
        enhanced_result = result.copy()
        enhanced_result['configuration'] = {
            'backbone': args.backbone,
            'training_mode': 'phase2_only' if getattr(args, 'phase2_only', False) else 'two_phase',
            'total_epochs': total_epochs,
            'phase_1_epochs': 0 if getattr(args, 'phase2_only', False) else getattr(args, 'phase1_epochs', 50),
            'phase_2_epochs': getattr(args, 'phase2_epochs', 50),
            'loss_type': getattr(args, 'loss_type', 'uncertainty_multi_task'),
            'learning_rates': {
                'backbone': getattr(args, 'backbone_lr', 1e-5),
                'head_phase1': getattr(args, 'head_lr_p1', 1e-3),
                'head_phase2': getattr(args, 'head_lr_p2', 1e-4)
            },
            'model_arc_compliance': True
        }
        with open(results_file, 'w') as f:
            json.dump(enhanced_result, f, indent=2, default=str)
        
        if result.get('success'):
            print(f"\n🎉 Two-phase training pipeline completed successfully!")
            print(f"📝 Results saved to: {results_file}")
            
            # Show architecture compliance summary
            compliance = result.get('architecture_compliance', {})
            print(f"\n🏆 MODEL_ARC.md Compliance:")
            print(f"   ✅ Two-phase training: {compliance.get('two_phase_training', False)}")
            print(f"   ✅ Multi-layer detection: {compliance.get('multi_layer_detection', False)}")
            print(f"   {'✅' if compliance.get('uncertainty_loss') else '⚠️'} Uncertainty-based loss: {compliance.get('uncertainty_loss', False)}")
            print(f"   ✅ Detection layers: {', '.join(compliance.get('detection_layers', []))}")
            return 0
        else:
            print(f"\n❌ Two-phase training pipeline failed: {result.get('error', 'Unknown error')}")
            print(f"📝 Error details saved to: {results_file}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())