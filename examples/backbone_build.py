#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/examples/backbone_build.py

Backbone build example with default parameters and model validation.
Builds both CSPDarkNet and EfficientNet-B4 backbones and runs validation API.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api.core import create_model_api, quick_build_model
from smartcash.model.api.backbone_api import check_data_prerequisites, check_built_models
from smartcash.ui.model.backbone.configs.backbone_defaults import get_default_backbone_config, get_available_backbones
from smartcash.common.logger import get_logger

# Setup logger - only warnings, errors, and phase transitions
logger = get_logger('backbone_build_example', level="WARNING")

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

def build_backbone_model(backbone_type: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Build a backbone model with default parameters.
    
    Args:
        backbone_type: Type of backbone to build ('efficientnet_b4' or 'cspdarknet')
        verbose: Enable verbose logging
        
    Returns:
        Dict containing build results
    """
    try:
        print(f"🏗️ Building {backbone_type} backbone...")
        
        # Create optimized progress callback
        progress_callback = create_optimized_progress_callback(verbose)
        
        # Get default configuration
        config = get_default_backbone_config()
        
        # Update backbone type in config
        config['backbone']['model_type'] = backbone_type
        config['model']['backbone'] = backbone_type
        
        # Ensure multi-layer configuration
        config['backbone']['layer_mode'] = 'multi'
        config['model']['layer_mode'] = 'multi'
        config['backbone']['detection_layers'] = ['layer_1', 'layer_2', 'layer_3']
        config['model']['detection_layers'] = ['layer_1', 'layer_2', 'layer_3']
        config['backbone']['multi_layer_heads'] = True
        config['model']['multi_layer_heads'] = True
        config['backbone']['num_classes'] = {
            'layer_1': 7,   # 7 denominations
            'layer_2': 7,   # 7 denomination-specific features  
            'layer_3': 3    # 3 common features
        }
        config['model']['num_classes'] = {
            'layer_1': 7,   # 7 denominations
            'layer_2': 7,   # 7 denomination-specific features  
            'layer_3': 3    # 3 common features
        }
        config['backbone']['feature_optimization'] = True
        
        # Log configuration if verbose
        if verbose:
            print(f"📋 Configuration:")
            print(f"   • Backbone: {backbone_type}")
            print(f"   • Layer mode: {config['backbone']['layer_mode']}")
            print(f"   • Pretrained: {config['backbone']['pretrained']}")
            print(f"   • Feature optimization: {config['backbone']['feature_optimization']}")
            print(f"   • Mixed precision: {config['backbone']['mixed_precision']}")
            print(f"   • Detection layers: {config['backbone']['detection_layers']}")
            print(f"   • Multi-layer heads: {config['backbone']['multi_layer_heads']}")
            print(f"   • Input size: {config['backbone']['input_size']}")
            print(f"   • Classes per layer: {config['backbone']['num_classes']}")
        
        # Build model using create_model_api for more control over configuration
        try:
            api = create_model_api(progress_callback=progress_callback)
            
            if api:
                # Build the model with full multi-layer configuration
                try:
                    # Build with the multi-layer configuration
                    model_config = {
                        'backbone': backbone_type,
                        'layer_mode': 'multi',
                        'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                        'multi_layer_heads': True,
                        'num_classes': {
                            'layer_1': 7,   # 7 denominations
                            'layer_2': 7,   # 7 denomination-specific features  
                            'layer_3': 3    # 3 common features
                        },
                        'pretrained': config['backbone']['pretrained'],
                        'feature_optimization': config['backbone']['feature_optimization'],
                        'mixed_precision': config['backbone']['mixed_precision'],
                        'img_size': config['backbone']['input_size']
                    }
                    
                    # Build the model - wrap config in 'model' key for proper nesting
                    model_info = api.build_model(model=model_config)
                    
                    if model_info.get('status') == 'built':
                        print(f"✅ {backbone_type} backbone built successfully!")
                        
                        # Save the initial built model to /data/models
                        try:
                            model_metrics = {
                                'backbone': backbone_type,
                                'parameters': model_info.get('total_parameters', 0),
                                'layer_mode': model_info.get('layer_mode', 'multi'),
                                'detection_layers': model_info.get('detection_layers', []),
                                'device': str(api.device),
                                'build_time': time.time()
                            }
                            
                            model_path = api.save_initial_model(
                                metrics=model_metrics,
                                model_name=f"{backbone_type}_backbone"
                            )
                            
                            if verbose:
                                print(f"💾 Initial model saved to: {model_path}")
                                
                        except Exception as save_error:
                            logger.warning(f"⚠️ Failed to save initial model: {str(save_error)}")
                            # Don't fail the build if save fails
                        
                        # Clean up progress bars
                        progress_callback.cleanup()
                        
                        return {
                            'success': True,
                            'backbone_type': backbone_type,
                            'model_info': model_info,
                            'device': str(api.device),
                            'parameters': model_info.get('total_parameters', 'N/A'),
                            'backbone_name': model_info.get('backbone', backbone_type),
                            'layer_mode': model_info.get('layer_mode', 'multi'),
                            'detection_layers': model_info.get('detection_layers', []),
                            'model_path': model_path if 'model_path' in locals() else None
                        }
                    else:
                        error_msg = model_info.get('message', 'Model not built properly')
                        logger.error(f"❌ Build failed for {backbone_type}: {error_msg}")
                        return {'success': False, 'backbone_type': backbone_type, 'error': error_msg}
                        
                except Exception as build_error:
                    # If build fails, catch the exception
                    error_msg = f"Build failed: {str(build_error)}"
                    logger.error(f"❌ {error_msg}")
                    return {'success': False, 'backbone_type': backbone_type, 'error': error_msg}
            else:
                error_msg = "Failed to create model API"
                logger.error(f"❌ {error_msg}")
                return {'success': False, 'backbone_type': backbone_type, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"❌ Exception during build: {str(e)}")
            return {'success': False, 'backbone_type': backbone_type, 'error': str(e)}
        finally:
            # Ensure cleanup
            try:
                progress_callback.cleanup()
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Failed to build {backbone_type}: {str(e)}")
        return {'success': False, 'backbone_type': backbone_type, 'error': str(e)}

def validate_model(backbone_type: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate built model using model validation API.
    
    Args:
        backbone_type: Type of backbone to validate
        verbose: Enable verbose logging
        
    Returns:
        Dict containing validation results
    """
    try:
        print(f"🔍 Validating {backbone_type} model...")
        
        # Check data prerequisites first
        prereq_result = check_data_prerequisites()
        if not prereq_result.get('prerequisites_ready'):
            logger.warning(f"⚠️ Data prerequisites not fully ready: {prereq_result.get('message')}")
            if verbose:
                print(f"   • Pretrained models: {prereq_result.get('pretrained_models', {}).get('available', False)}")
                print(f"   • Raw data: {prereq_result.get('raw_data', {}).get('available', False)}")
                print(f"   • Preprocessed data: {prereq_result.get('preprocessed_data', {}).get('available', False)}")
        
        # Check for built models
        models_result = check_built_models()
        if not models_result.get('success'):
            error_msg = f"Failed to check built models: {models_result.get('discovery_summary', 'Unknown error')}"
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'backbone_type': backbone_type, 'error': error_msg}
        
        # Look for models of the specified backbone type
        by_backbone = models_result.get('by_backbone', {})
        backbone_models = by_backbone.get(backbone_type, [])
        
        if not backbone_models:
            # Try normalized backbone name
            normalized_names = {
                'efficientnet_b4': ['efficientnet_b4', 'efficientnet'],
                'cspdarknet': ['cspdarknet', 'yolov5']
            }
            
            for alt_name in normalized_names.get(backbone_type, []):
                if alt_name in by_backbone:
                    backbone_models = by_backbone[alt_name]
                    break
        
        if backbone_models:
            latest_model = backbone_models[0]  # Models are sorted by timestamp
            print(f"✅ {backbone_type} model validation completed!")
            
            if verbose:
                print(f"   • Model path: {latest_model.get('path', 'N/A')}")
                print(f"   • Size: {latest_model.get('size_mb', 0):.1f} MB")
                print(f"   • Type: {latest_model.get('model_type', 'N/A')}")
                print(f"   • Accuracy: {latest_model.get('accuracy', 'N/A')}")
                print(f"   • Found in: {latest_model.get('found_in', 'N/A')}")
            
            return {
                'success': True,
                'backbone_type': backbone_type,
                'model_info': latest_model,
                'total_models': len(backbone_models),
                'validation_time': 'N/A'
            }
        else:
            error_msg = f"No built models found for {backbone_type}"
            print(f"⚠️ {error_msg}")
            
            # Show available models if verbose
            if verbose and by_backbone:
                print("   Available models:")
                for bb_type, models in by_backbone.items():
                    print(f"     • {bb_type}: {len(models)} model(s)")
            
            return {
                'success': False,
                'backbone_type': backbone_type,
                'error': error_msg,
                'available_backbones': list(by_backbone.keys())
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to validate {backbone_type}: {str(e)}")
        return {'success': False, 'backbone_type': backbone_type, 'error': str(e)}

def run_backbone_pipeline(backbone_types: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Run complete pipeline for multiple backbone types: build -> validate.
    
    Args:
        backbone_types: List of backbone types to build and validate
        verbose: Enable verbose logging
        
    Returns:
        Dict containing results for all backbones
    """
    results = {
        'total_backbones': len(backbone_types),
        'successful_builds': 0,
        'successful_validations': 0,
        'results': {}
    }
    
    for backbone_type in backbone_types:
        print(f"\n{'='*60}")
        print(f"🚀 Processing {backbone_type.upper()} Backbone")
        print(f"{'='*60}")
        
        # Step 1: Build backbone
        build_result = build_backbone_model(backbone_type, verbose)
        
        # Step 2: Validate model (always run, even if build failed)
        validation_result = validate_model(backbone_type, verbose)
        
        # Store results
        results['results'][backbone_type] = {
            'build': build_result,
            'validation': validation_result
        }
        
        # Update counters
        if build_result.get('success'):
            results['successful_builds'] += 1
        if validation_result.get('success'):
            results['successful_validations'] += 1
        
        # Show summary for this backbone
        build_status = "✅ SUCCESS" if build_result.get('success') else "❌ FAILED"
        validation_status = "✅ SUCCESS" if validation_result.get('success') else "❌ FAILED"
        
        print(f"\n📊 {backbone_type.upper()} Summary:")
        print(f"   • Build: {build_status}")
        print(f"   • Validation: {validation_status}")
        
        if build_result.get('success') and verbose:
            print(f"   • Device: {build_result.get('device', 'N/A')}")
            print(f"   • Parameters: {build_result.get('parameters', 'N/A')}")
        
        if validation_result.get('success') and verbose:
            model_info = validation_result.get('model_info', {})
            print(f"   • Model size: {model_info.get('size_mb', 0):.1f} MB")
            print(f"   • Available models: {validation_result.get('total_models', 0)}")
    
    return results

def main():
    """Main function to run backbone build and validation example."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build and validate backbone models')
    parser.add_argument('--backbones', type=str, 
                       default='efficientnet_b4,cspdarknet',
                       help='Comma-separated list of backbones to build (default: efficientnet_b4,cspdarknet)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (shows all messages)')
    parser.add_argument('--build-only', action='store_true',
                       help='Only build models, skip validation')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing models, skip building')
    
    args = parser.parse_args()
    
    # Configure logger based on verbose flag
    global logger
    if args.verbose:
        logger = get_logger('backbone_build_example', level="INFO")
    
    # Parse backbone types
    backbone_types = [b.strip() for b in args.backbones.split(',')]
    
    # Validate backbone types
    available_backbones = get_available_backbones()
    invalid_backbones = [b for b in backbone_types if b not in available_backbones]
    if invalid_backbones:
        logger.error(f"❌ Invalid backbone types: {invalid_backbones}")
        logger.error(f"Available types: {list(available_backbones.keys())}")
        return 1
    
    print("=" * 80)
    print("🏗️ SmartCash Backbone Build & Validation Example")
    print("=" * 80)
    print(f"🎯 Target backbones: {', '.join(backbone_types)}")
    print(f"📝 Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"🔧 Build only: {'Yes' if args.build_only else 'No'}")
    print(f"🔍 Validate only: {'Yes' if args.validate_only else 'No'}")
    print("-" * 80)
    
    try:
        if args.validate_only:
            # Only run validation
            total_successful = 0
            for backbone_type in backbone_types:
                result = validate_model(backbone_type, args.verbose)
                if result.get('success'):
                    total_successful += 1
            
            print(f"\n✨ Validation completed: {total_successful}/{len(backbone_types)} successful")
            return 0 if total_successful == len(backbone_types) else 1
        
        elif args.build_only:
            # Only run building
            total_successful = 0
            for backbone_type in backbone_types:
                result = build_backbone_model(backbone_type, args.verbose)
                if result.get('success'):
                    total_successful += 1
            
            print(f"\n✨ Building completed: {total_successful}/{len(backbone_types)} successful")
            return 0 if total_successful == len(backbone_types) else 1
        
        else:
            # Run complete pipeline
            results = run_backbone_pipeline(backbone_types, args.verbose)
            
            # Final summary
            print(f"\n{'='*80}")
            print("📊 Final Results Summary")
            print(f"{'='*80}")
            print(f"🎯 Total backbones processed: {results['total_backbones']}")
            print(f"🏗️ Successful builds: {results['successful_builds']}/{results['total_backbones']}")
            print(f"🔍 Successful validations: {results['successful_validations']}/{results['total_backbones']}")
            
            # Save detailed results to JSON
            results_file = 'backbone_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📝 Detailed results saved to: {results_file}")
            
            # Determine exit code
            all_successful = (results['successful_builds'] == results['total_backbones'] and 
                            results['successful_validations'] == results['total_backbones'])
            
            if all_successful:
                print("🎉 All operations completed successfully!")
                return 0
            else:
                print("⚠️ Some operations failed. Check the results above for details.")
                return 1
                
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())