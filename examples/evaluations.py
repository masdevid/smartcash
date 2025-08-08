#!/usr/bin/env python3
"""
SmartCash Model Evaluation - Simplified Interface

This example provides a streamlined evaluation interface that:
- Reuses training infrastructure for fast, consistent checkpoint loading
- Auto-detects best checkpoints for each backbone
- Eliminates redundant model building and warmup steps
- Provides simplified CLI focused on essential scenarios

Usage:
    # Evaluate all scenarios with best cspdarknet checkpoint
    python examples/evaluations.py --scenario-all --backbone cspdarknet
    
    # Evaluate all scenarios with best efficientnet_b4 checkpoint  
    python examples/evaluations.py --scenario-all --backbone efficientnet_b4
    
    # Evaluate specific scenario with specific checkpoint
    python examples/evaluations.py --scenario position_variation --checkpoint data/checkpoints/best_model.pt
    
    # List available backbones and checkpoints
    python examples/evaluations.py --list-resources
"""

# Fix OpenMP duplicate library issue before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.evaluation import (
    create_evaluation_service,
    create_checkpoint_selector,
    run_backbone_comparison_evaluation,
    generate_backbone_comparison_charts,
    validate_all_scenario_data_sources,
)
# Training infrastructure is imported within functions to avoid circular imports


def create_evaluation_config(args=None) -> Dict[str, Any]:
    """Create evaluation configuration with optional custom paths from arguments."""
    
    # Default paths
    test_dir = "data/preprocessed/test"
    evaluation_dir = "data/evaluation"
    results_dir = "data/evaluation/results"
    charts_dir = "data/evaluation/charts"
    logs_dir = "logs/validation_metrics"
    checkpoint_dir = "data/checkpoints"
    
    # Override with command line arguments if provided
    if args:
        test_dir = getattr(args, 'test_dir', test_dir)
        evaluation_dir = getattr(args, 'scenario_data_dir', evaluation_dir)
        logs_dir = getattr(args, 'logs_dir', logs_dir)
        checkpoint_dir = getattr(args, 'checkpoint_dir', checkpoint_dir)
    
    # Build discovery paths (include custom checkpoint dir first if different from default)
    discovery_paths = []
    if checkpoint_dir != "data/checkpoints":
        discovery_paths.append(checkpoint_dir)
    discovery_paths.extend([
        "data/checkpoints",
        "data/checkpoints/cspdarknet",
        "data/checkpoints/efficientnet_b4",
        "runs/train/*/weights",
        "models/checkpoints"
    ])
    
    return {
        "model": {
            "model_name": "smartcash_yolov5_integrated",
            "backbone": "cspdarknet",
            "pretrained": True,
            "layer_mode": "multi",
            "detection_layers": ["layer_1", "layer_2", "layer_3"],
            "num_classes": 17,  # Updated for hierarchical prediction (Layer 1: 0-6, Layer 2: 7-13, Layer 3: 14-16)
            "img_size": 640,
            "feature_optimization": {"enabled": True}
        },
        "evaluation": {
            "data": {
                "test_dir": test_dir,
                "evaluation_dir": evaluation_dir,
                "results_dir": results_dir,
                "charts_dir": charts_dir,
                "logs_dir": logs_dir
            },
            "checkpoints": {
                "discovery_paths": discovery_paths,
                "filename_patterns": [
                    "best_*.pt",
                    "last_*.pt", 
                    "*_best.pt",
                    "*/best_*.pt",  # Include subdirectories
                    "*/last_*.pt"   # Include subdirectories
                ],
                "required_keys": ["model_state_dict"],
                "supported_backbones": ["cspdarknet", "efficientnet_b4", "yolov5s", "unknown"],
                "min_val_map": 0.0,
                "sort_by": "val_map",
                "max_checkpoints": 10,
                "auto_select_best": True
            },
            "scenarios": {
                "position_variation": {
                    "name": "Position Variation",
                    "enabled": True,
                    "description": "Test model performance with various camera positions and angles",
                    "augmentation_config": {
                        "num_variations": 5,
                        "rotation_range": [-15, 15],
                        "translation_range": [-0.1, 0.1],
                        "scale_range": [0.8, 1.2]
                    }
                },
                "lighting_variation": {
                    "name": "Lighting Variation", 
                    "enabled": True,
                    "description": "Test model performance under different lighting conditions",
                    "augmentation_config": {
                        "num_variations": 5,
                        "brightness_range": [0.7, 1.3],
                        "contrast_range": [0.8, 1.2],
                        "gamma_range": [0.8, 1.2]
                    }
                }
            },
            "metrics": {
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "include_inference_timing": True,
                "calculate_per_class_metrics": True
            },
            "analysis": {
                "currency_analysis": {
                    "enabled": True,
                    "primary_layer": "banknote",
                    "confidence_threshold": 0.3
                },
                "class_analysis": {
                    "enabled": True
                }
            },
            "export": {
                "formats": ["json", "csv", "markdown"],
                "include_visualizations": True,
                "save_predictions": True
            }
        }
    }


def create_log_callback(verbose: bool = True):
    """Create a log callback for evaluation progress."""
    def log_callback(level: str, message: str, data: dict = None):
        """Handle log messages from the evaluation system."""
        level_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'debug': '🔍',
            'critical': '🚨'
        }
        
        icon = level_icons.get(level.lower(), '📝')
        print(f"{icon} [{level.upper()}] {message}")
        
        if verbose and data:
            for key, value in data.items():
                if key != 'message':
                    print(f"    {key}: {value}")
    
    return log_callback


def create_metrics_callback(verbose: bool = True):
    """Create a metrics callback for evaluation results."""
    def metrics_callback(results: Dict[str, Any]):
        """Handle metrics from evaluation results."""
        if not verbose:
            return
            
        print(f"\n📊 EVALUATION METRICS SUMMARY")
        print("=" * 50)
        
        if 'summary' in results:
            summary = results['summary']
            
            # Overall metrics
            if 'aggregated_metrics' in summary:
                metrics = summary['aggregated_metrics']
                print(f"🎯 Overall Performance:")
                
                if 'overall_metrics' in metrics:
                    overall = metrics['overall_metrics']
                    if 'mAP' in overall:
                        print(f"   mAP: {overall['mAP']:.3f}")
                    if 'precision' in overall:
                        print(f"   Precision: {overall['precision']:.3f}")
                    if 'recall' in overall:
                        print(f"   Recall: {overall['recall']:.3f}")
                    if 'f1_score' in overall:
                        print(f"   F1-Score: {overall['f1_score']:.3f}")
                
                # Best configurations
                if 'best_configurations' in metrics:
                    best = metrics['best_configurations']
                    print(f"\n🏆 Best Configurations:")
                    for metric, config in best.items():
                        if isinstance(config, dict) and 'backbone' in config:
                            print(f"   Best {metric}: {config['backbone']} ({config.get('value', 'N/A')})")
            
            # Key findings
            if 'key_findings' in summary:
                findings = summary['key_findings']
                if findings:
                    print(f"\n🔍 Key Findings:")
                    for finding in findings[:3]:  # Show top 3 findings
                        print(f"   • {finding}")
        
        # Scenario results
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            print(f"\n📋 Scenario Results:")
            for scenario, backbones in eval_results.items():
                print(f"   {scenario.replace('_', ' ').title()}:")
                for backbone, result in backbones.items():
                    metrics = result.get('metrics', {})
                    map_score = metrics.get('mAP', metrics.get('val_map', 0))
                    print(f"     {backbone}: mAP {map_score:.3f}")
    
    return metrics_callback


def create_progress_callback(verbose: bool = True):
    """Create a progress callback for evaluation tracking."""
    def progress_callback(progress_type: str, current: int, total: int, message: str = ""):
        """Handle progress updates from evaluation."""
        if not verbose:
            return
            
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r🔄 {progress_type.upper()}: |{bar}| {percentage:.1f}% {message}", end='')
        
        if current >= total:
            print()  # New line when complete
    
    return progress_callback


def find_best_checkpoint_for_backbone(backbone: str) -> str:
    """Find the best checkpoint for a given backbone, preferring _phase2 and latest date."""
    config = create_evaluation_config()
    checkpoint_selector = create_checkpoint_selector(config)
    
    # Get all available checkpoints
    available_checkpoints = checkpoint_selector.list_available_checkpoints()
    
    if not available_checkpoints:
        raise ValueError(f"No checkpoints found")
    
    # Filter by backbone
    backbone_checkpoints = [cp for cp in available_checkpoints 
                           if cp['backbone'].lower() == backbone.lower()]
    
    if not backbone_checkpoints:
        raise ValueError(f"No checkpoints found for backbone: {backbone}")
    
    # Prioritize checkpoints with _phase2 suffix and latest date
    def checkpoint_priority(cp):
        filename = cp['filename']
        
        # Priority scores (higher is better)
        phase2_score = 100 if '_phase2' in filename else 0
        
        # Extract date from filename for sorting (format: YYYYMMDD)
        import re
        date_match = re.search(r'_(\d{8})', filename)
        date_score = int(date_match.group(1)) if date_match else 0
        
        # Use mAP as tiebreaker
        map_score = cp.get('metrics', {}).get('val_map', 0) * 10  # Scale to make it significant
        
        return phase2_score + date_score + map_score
    
    # Sort by priority (highest first)
    backbone_checkpoints.sort(key=checkpoint_priority, reverse=True)
    
    best_checkpoint = backbone_checkpoints[0]
    print(f"📋 Priority selection: {best_checkpoint['filename']} (phase2: {'_phase2' in best_checkpoint['filename']})")
    
    return best_checkpoint['path']


def load_model_with_training_infrastructure(checkpoint_path: str) -> Any:
    """Load model using existing model API infrastructure for consistency."""
    print(f"📦 Loading model: {Path(checkpoint_path).name}")
    
    # Import here to avoid circular imports
    from smartcash.model.api.core import create_api
    import torch
    
    try:
        # Load checkpoint to extract model configuration
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Use configuration from checkpoint if available
        model_config = checkpoint_data.get('config', {})
        if not model_config:
            # Fallback config based on checkpoint analysis
            state_dict = checkpoint_data.get('model_state_dict', {})
            detection_head_keys = [k for k in state_dict.keys() if '.m.0.weight' in k and ('head' in k or 'model.24' in k)]
            
            if detection_head_keys:
                output_channels = state_dict[detection_head_keys[0]].shape[0]
                num_classes = (output_channels // 3) - 5  # YOLOv5 formula: (classes + 5) * 3 anchors
                
                model_config = {
                    'backbone': 'cspdarknet',
                    'num_classes': num_classes,
                    'layer_mode': 'multi',
                    'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                    'pretrained': False,
                    'img_size': 640
                }
                print(f"🔍 Inferred model config: {num_classes} classes, {output_channels} outputs")
        
        # Create model API with checkpoint configuration
        config = {
            'model': model_config
        }
        api = create_api(config=config, use_yolov5_integration=True)
        
        # Build model with correct configuration first
        build_result = api.build_model(model_config=model_config)
        if not build_result.get('success', False):
            print(f"❌ Failed to build model: {build_result.get('error')}")
            return None
        
        # Load checkpoint
        result = api.load_checkpoint(checkpoint_path)
        
        if result.get('success', False):
            print(f"✅ Model loaded: {result.get('message', 'Success')}")
            return api
        else:
            print(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def run_single_scenario(args) -> Dict[str, Any]:
    """Run evaluation on a single scenario using training infrastructure."""
    print(f"🎯 Running scenario evaluation: {args.scenario}")
    
    if not args.checkpoint:
        print("❌ Error: --checkpoint is required for single scenario evaluation")
        return {'status': 'error', 'error': 'Checkpoint path required'}
    
    try:
        # Load model using training infrastructure (no warmup needed)
        model_api = load_model_with_training_infrastructure(args.checkpoint)
        
        if not model_api:
            return {'status': 'error', 'error': 'Failed to load model'}
        
        # Create simple evaluation service
        config = create_evaluation_config(args)
        service = create_evaluation_service(model_api=model_api, config=config)
        
        # Run scenario evaluation
        result = service.run_scenario(args.scenario, args.checkpoint)
        
        if result['status'] == 'success':
            _print_evaluation_results(args.scenario, result['metrics'])
        else:
            print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def run_all_scenarios(args) -> Dict[str, Any]:
    """Run evaluation on all scenarios with best checkpoint for backbone(s)."""
    
    # Determine backbones to evaluate
    backbones = [args.backbone] if args.backbone else None  # None triggers default comparison
    scenarios = ['position_variation', 'lighting_variation']
    
    # Create configuration
    config = create_evaluation_config(args)
    
    # Validate scenario data sources first
    validation_result = validate_all_scenario_data_sources(config)
    print(f"📁 Data source validation: {validation_result['scenarios_valid']}/{validation_result['scenarios_configured']} scenarios valid")
    
    if validation_result['scenarios_invalid'] > 0:
        print(f"⚠️ Warning: {validation_result['scenarios_invalid']} scenarios have invalid data sources")
        for issue in validation_result['issues'][:3]:  # Show first 3 issues
            print(f"   • {issue}")
    
    # Run backbone comparison evaluation
    print("🚀 Starting backbone comparison evaluation...")
    result = run_backbone_comparison_evaluation(
        backbones=backbones,
        scenarios=scenarios, 
        config=config,
        verbose=args.verbose
    )
    
    # Generate comparison charts
    chart_files = []
    charts_dir = getattr(args, 'charts_output_dir', 'data/evaluation/charts')
    
    if result['status'] == 'success':
        print("📊 Generating comparison charts...")
        try:
            chart_files = generate_backbone_comparison_charts(result, charts_dir)
            print(f"✅ Generated {len(chart_files)} comparison charts")
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Add chart files to result
    result['chart_files'] = chart_files
    
    # Print summary using the manager's built-in summary printer
    if result['status'] == 'success':
        from smartcash.model.evaluation.managers.backbone_comparison_manager import create_backbone_comparison_manager
        manager = create_backbone_comparison_manager(config)
        manager.print_comparison_summary(result)
    
    return result


def _print_evaluation_results(scenario: str, metrics: Dict[str, Any], brief: bool = False) -> None:
    """Print evaluation results in a consistent format."""
    if brief:
        map_score = metrics.get('map50', 0)
        accuracy = metrics.get('accuracy', 0)
        print(f"   Results: mAP={map_score:.3f}, Accuracy={accuracy:.3f}")
        return
    
    print(f"\n📊 Results for {scenario}:")
    print(f"\n🎯 Object Detection:")
    print(f"   mAP@50: {metrics.get('map50', 0):.3f}")
    print(f"   Precision: {metrics.get('precision', 0):.3f}")
    print(f"   Recall: {metrics.get('recall', 0):.3f}")
    print(f"   F1: {metrics.get('f1', 0):.3f}")
    
    print(f"\n💰 Classification:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
    
    if 'inference_time_avg' in metrics:
        print(f"\n⏱️ Performance:")
        print(f"   Inference: {metrics['inference_time_avg']:.3f}s")
        print(f"   FPS: {metrics.get('fps', 0):.1f}")


def list_available_resources(args) -> None:
    """List available checkpoints and backbones."""
    print(f"📋 Available Resources")
    print("=" * 40)
    
    config = create_evaluation_config(args)
    
    # List checkpoints grouped by backbone
    print(f"\n🏷️ Available Checkpoints:")
    checkpoint_selector = create_checkpoint_selector(config)
    checkpoints = checkpoint_selector.list_available_checkpoints()
    
    if checkpoints:
        # Group by backbone
        backbone_checkpoints = {}
        for cp in checkpoints:
            backbone = cp['backbone']
            if backbone not in backbone_checkpoints:
                backbone_checkpoints[backbone] = []
            backbone_checkpoints[backbone].append(cp)
        
        for backbone, cps in backbone_checkpoints.items():
            print(f"\n   🏗️ {backbone.upper()}:")
            for cp in cps[:3]:  # Show top 3 per backbone
                metrics = cp['metrics']
                map_score = metrics.get('val_map', metrics.get('mAP', 0))
                print(f"      • {Path(cp['path']).name} (mAP: {map_score:.3f})")
            if len(cps) > 3:
                print(f"      ... and {len(cps) - 3} more")
        
        # Show best for each backbone
        print(f"\n🏆 Best Checkpoints:")
        for backbone in backbone_checkpoints:
            best_path = find_best_checkpoint_for_backbone(backbone)
            if best_path:
                print(f"   {backbone}: {Path(best_path).name}")
    else:
        print("   No checkpoints found. Train models first.")
    
    # List available scenarios
    print(f"\n🎯 Available Scenarios:")
    print(f"   • position_variation")
    print(f"   • lighting_variation")




def create_argument_parser():
    """Create simplified command line argument parser."""
    parser = argparse.ArgumentParser(
        description='SmartCash Model Evaluation System - Simplified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all scenarios with best cspdarknet checkpoint
  python examples/evaluations.py --scenario-all --backbone cspdarknet
  
  # Evaluate all scenarios with best efficientnet_b4 checkpoint
  python examples/evaluations.py --scenario-all --backbone efficientnet_b4
  
  # Evaluate specific scenario with specific checkpoint
  python examples/evaluations.py --scenario position_variation --checkpoint data/checkpoints/best_model.pt
  
  # List available backbones and checkpoints
  python examples/evaluations.py --list-resources
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--scenario', type=str, 
                           choices=['position_variation', 'lighting_variation'],
                           help='Run single scenario evaluation (requires --checkpoint)')
    mode_group.add_argument('--scenario-all', action='store_true',
                           help='Run evaluation on all scenarios (optional --backbone, defaults to comparison)')
    mode_group.add_argument('--list-resources', action='store_true',
                           help='List available checkpoints and scenarios')
    
    # Parameters for specific modes
    parser.add_argument('--backbone', type=str,
                       choices=['cspdarknet', 'efficientnet_b4'],
                       help='Backbone architecture (optional for --scenario-all, defaults to comparison)')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to specific checkpoint file (required for --scenario)')
    
    # Optional parameters
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save evaluation results')
    parser.add_argument('--test-dir', type=str, default='data/preprocessed/test',
                       help='Directory containing test images (default: data/preprocessed/test)')
    
    return parser


def main():
    """Main function for evaluation example."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🚀 SmartCash Model Evaluation")
    print("=" * 40)
    
    try:
        # Route to appropriate function based on mode
        if args.scenario:
            if not args.checkpoint:
                print("❌ Error: --checkpoint is required for single scenario evaluation")
                return 1
            result = run_single_scenario(args)
            
        elif args.scenario_all:
            # Backbone is now optional - defaults to comparison if not specified
            result = run_all_scenarios(args)
            
        elif args.list_resources:
            list_available_resources(args)
            return 0
        
        # Check result status
        if isinstance(result, dict):
            if result.get('status') == 'success':
                print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY")
                return 0
            else:
                print(f"\n❌ EVALUATION FAILED")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ EVALUATION INTERRUPTED BY USER")
        return 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())