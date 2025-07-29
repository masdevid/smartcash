#!/usr/bin/env python3
"""
SmartCash Model Evaluation Example

This example demonstrates how to use the SmartCash evaluation system to:
- Evaluate trained models on different research scenarios
- Compare performance across different backbones
- Generate comprehensive evaluation reports
- Test models with position and lighting variations

Usage:
    # Single scenario with specific checkpoint
    python examples/evaluations.py --scenario position_variation --checkpoint data/checkpoints/best_model.pt
    
    # All scenarios with top 3 checkpoints
    python examples/evaluations.py --all-scenarios --checkpoint-dir data/checkpoints --top-n 3 --verbose
    
    # Evaluate specific checkpoints by name
    python examples/evaluations.py --all-scenarios --checkpoints "best_cspdarknet.pt,best_efficientnet.pt"
    
    # Filter by backbone and minimum performance
    python examples/evaluations.py --all-scenarios --backbone cspdarknet --min-map 0.5
    
    # Compare different backbone architectures
    python examples/evaluations.py --compare-backbones --checkpoint-dir data/checkpoints
    
Features:
    - Multiple research scenarios (position, lighting variations)
    - Automatic checkpoint discovery and selection
    - Comprehensive metrics calculation (mAP, precision, recall, F1)
    - Performance benchmarking with inference timing
    - Export results in multiple formats (JSON, CSV, Markdown)
"""

# Fix OpenMP duplicate library issue before any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.evaluation import (
    create_evaluation_service,
    run_evaluation_pipeline,
    create_checkpoint_selector,
    create_scenario_manager
)
from smartcash.model.api.core import create_api


def create_evaluation_config() -> Dict[str, Any]:
    """Create default evaluation configuration."""
    return {
        "evaluation": {
            "data": {
                "test_dir": "data/preprocessed/test",
                "evaluation_dir": "data/evaluation",
                "results_dir": "data/evaluation/results"
            },
            "checkpoints": {
                "discovery_paths": [
                    "data/checkpoints",
                    "runs/train/*/weights",
                    "models/checkpoints"
                ],
                "filename_patterns": [
                    "best_*.pt",
                    "last.pt",
                    "*_best.pt"
                ],
                "required_keys": ["model_state_dict"],
                "supported_backbones": ["cspdarknet", "efficientnet_b4", "yolov5s", "unknown"],
                "min_val_map": 0.1,
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
                        "translation_range": 0.1,
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


def get_selected_checkpoints(args, config: Dict[str, Any]):
    """Get list of checkpoints based on user selection criteria."""
    checkpoint_selector = create_checkpoint_selector(config)
    
    # Set discovery paths
    if args.checkpoint_dir:
        checkpoint_selector.discovery_paths = [Path(args.checkpoint_dir)]
    
    # Get all available checkpoints
    available_checkpoints = checkpoint_selector.list_available_checkpoints()
    
    if not available_checkpoints:
        return []
    
    # If specific checkpoint provided, use it
    if args.checkpoint:
        return [args.checkpoint]
    
    # If specific checkpoints listed, find them
    if args.checkpoints:
        selected_checkpoints = []
        checkpoint_names = [name.strip() for name in args.checkpoints.split(',')]
        
        for cp in available_checkpoints:
            cp_name = Path(cp['path']).name
            # Match by filename or display name
            if (cp_name in checkpoint_names or 
                cp['display_name'] in checkpoint_names or
                any(name in cp_name for name in checkpoint_names)):
                selected_checkpoints.append(cp['path'])
        
        if not selected_checkpoints:
            print(f"⚠️ Warning: No checkpoints found matching: {', '.join(checkpoint_names)}")
            print(f"Available checkpoints:")
            for cp in available_checkpoints[:5]:
                print(f"   - {Path(cp['path']).name}")
        
        return selected_checkpoints
    
    # Filter by backbone if specified
    filtered_checkpoints = available_checkpoints
    if args.backbone:
        filtered_checkpoints = [cp for cp in filtered_checkpoints 
                              if cp['backbone'].lower() == args.backbone.lower()]
        
        if not filtered_checkpoints:
            print(f"⚠️ Warning: No checkpoints found for backbone '{args.backbone}'")
            available_backbones = list(set(cp['backbone'] for cp in available_checkpoints))
            print(f"Available backbones: {', '.join(available_backbones)}")
            return []
    
    # Filter by minimum mAP if specified
    if args.min_map is not None:
        filtered_checkpoints = [cp for cp in filtered_checkpoints 
                              if cp.get('metrics', {}).get('val_map', 0) >= args.min_map]
        
        if not filtered_checkpoints:
            print(f"⚠️ Warning: No checkpoints found with mAP >= {args.min_map}")
    
    # Take top N checkpoints
    selected = filtered_checkpoints[:args.top_n]
    return [cp['path'] for cp in selected]


def run_single_scenario(args) -> Dict[str, Any]:
    """Run evaluation on a single scenario."""
    print(f"🎯 Running single scenario evaluation: {args.scenario}")
    
    config = create_evaluation_config()
    
    # Create model API if needed
    model_api = None
    if args.checkpoint:
        try:
            print(f"🔧 Creating model API for checkpoint: {args.checkpoint}")
            model_api = create_api(
                config=config,
                use_yolov5_integration=True
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not create model API: {e}")
    
    # Create evaluation service
    service = create_evaluation_service(model_api=model_api, config=config)
    
    # Run scenario evaluation
    result = service.run_scenario(args.scenario, args.checkpoint)
    
    if result['status'] == 'success':
        print(f"✅ Scenario evaluation completed successfully!")
        
        metrics = result['metrics']
        print(f"\n📊 Results for {args.scenario}:")
        print(f"   mAP: {metrics.get('mAP', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")
        print(f"   F1-Score: {metrics.get('f1_score', 0):.3f}")
        
        if 'inference_time_avg' in metrics:
            print(f"   Avg Inference Time: {metrics['inference_time_avg']:.3f}s")
        
    else:
        print(f"❌ Scenario evaluation failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_all_scenarios(args) -> Dict[str, Any]:
    """Run evaluation on all enabled scenarios."""
    print(f"🚀 Running comprehensive evaluation on all scenarios")
    
    config = create_evaluation_config()
    
    # Determine checkpoints to use
    checkpoints = get_selected_checkpoints(args, config)
    if not checkpoints:
        print("❌ No checkpoints found to evaluate")
        return {'status': 'error', 'error': 'No checkpoints found'}
    
    print(f"📋 Selected {len(checkpoints)} checkpoint(s) for evaluation:")
    for i, cp_path in enumerate(checkpoints, 1):
        print(f"   {i}. {Path(cp_path).name}")
    
    # Create callbacks
    metrics_callback = create_metrics_callback(args.verbose)
    progress_callback = create_progress_callback(args.verbose)
    
    # Run comprehensive evaluation
    result = run_evaluation_pipeline(
        scenarios=None,  # Use all enabled scenarios
        checkpoints=checkpoints,
        model_api=None,  # Let the service create the API
        config=config,
        progress_callback=progress_callback,
        ui_components={}
    )
    
    if result['status'] == 'success':
        print(f"\n✅ Comprehensive evaluation completed!")
        print(f"   Scenarios evaluated: {result['scenarios_evaluated']}")
        print(f"   Checkpoints evaluated: {result['checkpoints_evaluated']}")
        
        # Call metrics callback to display results
        metrics_callback(result)
        
        # Save results if requested
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            service = create_evaluation_service(config=config)
            export_files = service.save_results(result, formats=['json', 'csv', 'markdown'])
            
            print(f"\n💾 Results saved to:")
            for format_name, file_path in export_files.items():
                print(f"   {format_name.upper()}: {file_path}")
        
    else:
        print(f"❌ Comprehensive evaluation failed: {result.get('error', 'Unknown error')}")
    
    return result


def compare_backbones(args) -> Dict[str, Any]:
    """Compare performance across different backbones."""
    print(f"⚖️ Comparing backbone performance")
    
    config = create_evaluation_config()
    
    # Get available checkpoints
    checkpoint_selector = create_checkpoint_selector(config)
    if args.checkpoint_dir:
        checkpoint_selector.discovery_paths = [Path(args.checkpoint_dir)]
    
    available_checkpoints = checkpoint_selector.list_available_checkpoints()
    
    # Group by backbone
    backbone_checkpoints = {}
    checkpoints_to_use = available_checkpoints[:args.top_n * 3]  # Get more to ensure coverage
    
    for cp in checkpoints_to_use:
        backbone = cp['backbone']
        if backbone not in backbone_checkpoints:
            backbone_checkpoints[backbone] = []
        backbone_checkpoints[backbone].append(cp)
    
    # Take best checkpoint for each backbone
    backbone_best = {}
    for backbone, cps in backbone_checkpoints.items():
        # Sort by mAP and take the best
        cps.sort(key=lambda x: x.get('metrics', {}).get('val_map', 0), reverse=True)
        backbone_best[backbone] = cps[0]['path']
    
    print(f"📊 Found checkpoints for backbones: {list(backbone_best.keys())}")
    
    # Run evaluation for each backbone
    results = {}
    for backbone, checkpoint_path in backbone_best.items():
        print(f"\n🏗️ Evaluating {backbone} backbone...")
        print(f"   Using checkpoint: {Path(checkpoint_path).name}")
        
        result = run_evaluation_pipeline(
            scenarios=['position_variation', 'lighting_variation'],
            checkpoints=[checkpoint_path],
            config=config,
            progress_callback=create_progress_callback(args.verbose)
        )
        
        results[backbone] = result
    
    # Display comparison
    print(f"\n📊 BACKBONE COMPARISON RESULTS")
    print("=" * 60)
    
    for backbone, result in results.items():
        if result['status'] == 'success':
            summary = result.get('summary', {})
            metrics = summary.get('aggregated_metrics', {}).get('overall_metrics', {})
            
            print(f"\n🏗️ {backbone.upper()}:")
            print(f"   mAP: {metrics.get('mAP', 0):.3f}")
            print(f"   Precision: {metrics.get('precision', 0):.3f}")
            print(f"   Recall: {metrics.get('recall', 0):.3f}")
            print(f"   F1-Score: {metrics.get('f1_score', 0):.3f}")
            
            if 'inference_time_avg' in metrics:
                print(f"   Avg Inference Time: {metrics['inference_time_avg']:.3f}s")
        else:
            print(f"\n🏗️ {backbone.upper()}: ❌ Failed")
    
    return results


def list_available_resources(args) -> None:
    """List available checkpoints and scenarios."""
    print(f"📋 Available Resources")
    print("=" * 40)
    
    config = create_evaluation_config()
    
    # List checkpoints
    print(f"\n🏷️ Available Checkpoints:")
    checkpoint_selector = create_checkpoint_selector(config)
    if args.checkpoint_dir:
        checkpoint_selector.discovery_paths = [Path(args.checkpoint_dir)]
    
    checkpoints = checkpoint_selector.list_available_checkpoints()
    
    if checkpoints:
        for i, cp in enumerate(checkpoints, 1):
            metrics = cp['metrics']
            map_score = metrics.get('val_map', metrics.get('mAP', 0))
            print(f"   {i}. {cp['display_name']}")
            print(f"      Path: {cp['path']}")
            print(f"      Backbone: {cp['backbone']}")
            print(f"      mAP: {map_score:.3f}")
            print(f"      Size: {cp['file_size_mb']} MB")
            print()
    else:
        print("   No checkpoints found. Make sure to:")
        print("   - Train a model first using examples/callback_only_training_example.py")
        print("   - Check that checkpoint paths exist")
        print("   - Verify checkpoint format compatibility")
    
    # List scenarios
    print(f"\n🎯 Available Scenarios:")
    scenario_manager = create_scenario_manager(config)
    scenarios = scenario_manager.list_available_scenarios()
    
    for scenario in scenarios:
        status_icon = "✅" if scenario.get('ready', False) else "⚠️"
        print(f"   {status_icon} {scenario['display_name']}")
        print(f"      Name: {scenario['name']}")
        print(f"      Enabled: {scenario['enabled']}")
        print(f"      Data exists: {scenario['data_exists']}")
        if 'description' in scenario:
            print(f"      Description: {scenario['description']}")
        print()


def setup_scenarios(args) -> None:
    """Setup and prepare evaluation scenarios."""
    print(f"🚀 Setting up evaluation scenarios")
    
    config = create_evaluation_config()
    scenario_manager = create_scenario_manager(config)
    
    # Prepare all scenarios
    result = scenario_manager.prepare_all_scenarios(force_regenerate=args.force_regenerate)
    
    print(f"\n📊 Scenario Setup Results:")
    print(f"   Total scenarios: {result['total_scenarios']}")
    print(f"   Successful: {result['successful']}")
    print(f"   Failed: {result['failed']}")
    
    # Display detailed results
    for scenario_name, scenario_result in result['results'].items():
        if scenario_result['status'] == 'successful' or scenario_result['status'] == 'existing':
            print(f"   ✅ {scenario_name}: Ready")
            if 'validation' in scenario_result:
                validation = scenario_result['validation']
                print(f"      Images: {validation.get('images_count', 0)}")
                print(f"      Labels: {validation.get('labels_count', 0)}")
        else:
            print(f"   ❌ {scenario_name}: {scenario_result.get('error', 'Failed')}")


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='SmartCash Model Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single scenario with specific checkpoint
  python examples/evaluations.py --scenario position_variation --checkpoint data/checkpoints/best_model.pt
  
  # Run comprehensive evaluation on all scenarios with top 3 checkpoints
  python examples/evaluations.py --all-scenarios --checkpoint-dir data/checkpoints --top-n 3 --verbose
  
  # Evaluate specific checkpoints by name
  python examples/evaluations.py --all-scenarios --checkpoints "best_cspdarknet_multi.pt,best_efficientnet_b4.pt"
  
  # Filter checkpoints by backbone and minimum mAP
  python examples/evaluations.py --all-scenarios --backbone cspdarknet --min-map 0.5 --top-n 2
  
  # Compare different backbones (automatically selects best checkpoint per backbone)
  python examples/evaluations.py --compare-backbones --checkpoint-dir data/checkpoints --top-n 5
  
  # List available resources to see what checkpoints and scenarios exist
  python examples/evaluations.py --list-resources --checkpoint-dir data/checkpoints
  
  # Setup evaluation scenarios (required before first evaluation)
  python examples/evaluations.py --setup-scenarios --force-regenerate
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--scenario', type=str, 
                           choices=['position_variation', 'lighting_variation'],
                           help='Run single scenario evaluation')
    mode_group.add_argument('--all-scenarios', action='store_true',
                           help='Run evaluation on all enabled scenarios')
    mode_group.add_argument('--compare-backbones', action='store_true',
                           help='Compare performance across different backbones')
    mode_group.add_argument('--list-resources', action='store_true',
                           help='List available checkpoints and scenarios')
    mode_group.add_argument('--setup-scenarios', action='store_true',
                           help='Setup and prepare evaluation scenarios')
    
    # Checkpoint selection options
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--checkpoint', type=str,
                                 help='Path to specific checkpoint file')
    checkpoint_group.add_argument('--checkpoints', type=str,
                                 help='Comma-separated list of checkpoint names/patterns to evaluate')
    
    # Checkpoint discovery and filtering
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                       help='Directory to search for checkpoints (default: data/checkpoints)')
    parser.add_argument('--backbone', type=str,
                       choices=['cspdarknet', 'efficientnet_b4', 'yolov5s'],
                       help='Filter checkpoints by backbone architecture')
    parser.add_argument('--min-map', type=float,
                       help='Minimum mAP threshold for checkpoint selection')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of top checkpoints to evaluate (default: 3)')
    
    # General options
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save evaluation results')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of scenario data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser


def main():
    """Main function for evaluation example."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🚀 STARTING SmartCash Model Evaluation")
    print("=" * 50)
    
    try:
        # Route to appropriate function based on mode
        if args.scenario:
            if not args.checkpoint:
                print("❌ Error: --checkpoint is required for single scenario evaluation")
                return 1
            result = run_single_scenario(args)
            
        elif args.all_scenarios:
            result = run_all_scenarios(args)
            
        elif args.compare_backbones:
            result = compare_backbones(args)
            
        elif args.list_resources:
            list_available_resources(args)
            return 0
            
        elif args.setup_scenarios:
            setup_scenarios(args)
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