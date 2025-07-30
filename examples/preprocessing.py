"""
File: smartcash/examples/preprocessing.py
Description: Optimized preprocessing API example with cleanup support and minimal logging
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.api.preprocessing_api import preprocess_dataset, get_default_config
from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files

# Setup logger - only warnings, errors, and phase transitions
logger = get_logger("preprocessing_example", level="WARNING")

def setup_arg_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Preprocess dataset example')
    
    # Dataset arguments
    parser.add_argument('--data-dir', type=str, 
                      default=os.getenv('DATA_DIR', 'data/'),
                      help='Base path to dataset directory')
    parser.add_argument('--output-dir', type=str, 
                      default=os.getenv('PREPROCESSED_DIR', 'data/preprocessed'),
                      help='Path to save preprocessed dataset')
    parser.add_argument('--splits', type=str, nargs='+', 
                      default=os.getenv('SPLITS', 'train valid test').split(),
                      help='Dataset splits to process (space separated)')
    
    # Normalization arguments
    parser.add_argument('--normalization', type=str, 
                      default=os.getenv('NORMALIZATION_PRESET', 'default'),
                      choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'inference', 'default'],
                      help='Normalization preset to use (default: default)')
    parser.add_argument('--augment', action='store_true',
                      default=os.getenv('AUGMENT', 'False').lower() == 'true',
                      help='Enable data augmentation')
    
    # Performance options
    parser.add_argument('--workers', type=int, 
                      default=int(os.getenv('WORKERS', '8')),
                      help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, 
                      default=int(os.getenv('BATCH_SIZE', '32')),
                      help='Batch size for processing')
    
    # Cleanup and logging options
    parser.add_argument('--cleanup', action='store_true', default=True,
                      help='Clean up existing preprocessed data before starting (default: True)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging (shows all messages)')
    
    return parser

def create_preprocessing_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create preprocessing configuration from command line arguments"""
    # Get default config from the API
    from smartcash.dataset.preprocessor.config.defaults import get_default_config
    config = get_default_config()
    
    # Update base paths
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # Ensure preprocessing exists in config
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    # Update target splits
    config['preprocessing']['target_splits'] = args.splits
    
    # Set up normalization
    if 'normalization' not in config:
        config['normalization'] = {}
    
    # Update normalization preset if specified
    if args.normalization in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'inference', 'default']:
        config['normalization'].update({
            'preset': args.normalization,
            'enabled': True
        })
    
    # Set up augmentation if enabled
    if args.augment:
        if 'augmentation' not in config:
            config['augmentation'] = {}
        config['augmentation']['enabled'] = True
    
    # Update performance settings
    if 'performance' not in config:
        config['performance'] = {}
    config['performance'].update({
        'num_workers': args.workers,
        'batch_size': args.batch_size
    })
    
    return config

def create_optimized_progress_callback(verbose: bool = False) -> Callable:
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

def main() -> int:
    """Main function to run preprocessing"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Configure logger based on verbose flag
    global logger
    if args.verbose:
        logger = get_logger("preprocessing_example", level="INFO")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up existing preprocessed data if requested
    if args.cleanup:
        print("🧹 Cleaning up existing preprocessed data...")
        try:
            cleanup_result = cleanup_preprocessing_files(
                data_dir=args.output_dir, 
                target='preprocessed',
                splits=args.splits,  # Target all specified splits
                confirm=True
            )
            if cleanup_result.get('success'):
                print("✅ Cleanup completed successfully")
                if args.verbose and 'by_split' in cleanup_result:
                    for split, stats in cleanup_result['by_split'].items():
                        print(f"   • {split}: {stats.get('files_removed', 0)} files removed")
            else:
                logger.warning(f"⚠️ Cleanup warning: {cleanup_result.get('message', 'Unknown issue')}")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 1
    
    # Create preprocessing configuration
    config = create_preprocessing_config(args)
    
    # Display configuration
    print("=" * 60)
    print("🚀 SmartCash Dataset Preprocessing")
    print("=" * 60)
    print(f"📂 Input directory: {args.data_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🔢 Splits: {', '.join(args.splits)}")
    print(f"🔄 Normalization: {args.normalization}")
    print(f"✨ Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print(f"⚡ Workers: {args.workers}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🧹 Cleanup mode: {'Enabled' if args.cleanup else 'Disabled'}")
    print(f"🔍 Sample validation: {'Enabled' if config.get('preprocessing', {}).get('validation', {}).get('enabled', True) else 'Disabled'}")
    print(f"🗂️ Invalid sample quarantine: Enabled (data/invalid/)")
    print(f"📝 Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print("-" * 60)
    
    # Show validation info if enabled
    if config.get('preprocessing', {}).get('validation', {}).get('enabled', True):
        print("🔍 Sample Validation Info:")
        print("   • Validates bounding box format and size")
        print("   • Enforces hierarchical class relationships:")
        print("     - Multi-label samples: ignore Layer 3 (l3_*) classes")
        print("     - If Layer 2 (l2_*) exists: must match with Layer 1")
        print("     - Single labels valid: either 001 OR l2_001")
        print("   • Auto-fix capability for invalid label combinations:")
        print("     - Example: 001, l2_001, l2_100 → auto-fix to 001, l2_001")
        print("   • Invalid samples moved to data/invalid/ directory")
        print("   • Only valid samples proceed to preprocessing")
        print("-" * 60)
    
    try:
        # Set up optimized progress callback
        progress_callback = create_optimized_progress_callback(args.verbose)
        
        # Run preprocessing
        print("🚀 Starting preprocessing...")
        results = preprocess_dataset(
            config=config,
            progress_callback=progress_callback,
            splits=args.splits
        )
        
        # Close any remaining progress bars
        progress_callback.cleanup()
        
        # Display results with summary
        print(f"\n✅ Preprocessing completed successfully!")
        print("📊 Processing Summary:")
        print(f"   • Total splits processed: {len(args.splits)}")
        
        # Show per-split statistics if available
        stats = results.get('stats', {})
        if 'by_split' in stats:
            for split, split_stats in stats['by_split'].items():
                print(f"   • {split.upper()} Split:")
                print(f"     - Processed: {split_stats.get('processed', 0)}")
                print(f"     - Errors: {split_stats.get('errors', 0)}")
                print(f"     - Quarantined: {split_stats.get('quarantined', 0)}")
                print(f"     - Total: {split_stats.get('total', 0)}")
        
        # Show overall statistics
        print("\n📈 Overall Statistics:")
        input_stats = stats.get('input', {})
        output_stats = stats.get('output', {})
        performance_stats = stats.get('performance', {})
        validation_stats = stats.get('validation', {})
        
        print(f"   • Total input images: {input_stats.get('total_images', 0)}")
        print(f"   • Total processed: {output_stats.get('total_processed', 0)}")
        print(f"   • Total errors: {output_stats.get('total_errors', 0)}")
        print(f"   • Total quarantined: {output_stats.get('total_quarantined', 0)}")
        print(f"   • Success rate: {output_stats.get('success_rate', 'N/A')}")
        print(f"   • Quarantine rate: {output_stats.get('quarantine_rate', 'N/A')}")
        print(f"   • Processing speed: {performance_stats.get('images_per_second', 0)} images/sec")
        
        # Show validation statistics if available
        if validation_stats and validation_stats.get('enabled'):
            print(f"\n🔍 Sample Validation Statistics:")
            print(f"   • Samples validated: {validation_stats.get('samples_validated', 0)}")
            print(f"   • Valid samples: {validation_stats.get('valid_samples', 0)}")
            print(f"   • Invalid samples: {validation_stats.get('invalid_samples', 0)}")
            print(f"   • Auto-fixed samples: {validation_stats.get('auto_fixed_samples', 0)}")
            print(f"   • Quarantined samples: {validation_stats.get('quarantined_samples', 0)}")
            
            # Show auto-fix statistics if any
            auto_fix_stats = validation_stats.get('auto_fix_stats', {})
            if validation_stats.get('auto_fixed_samples', 0) > 0:
                print(f"   • Auto-fix breakdown:")
                if auto_fix_stats.get('layer_3_removed', 0) > 0:
                    print(f"     - Layer_3 labels removed: {auto_fix_stats['layer_3_removed']}")
                if auto_fix_stats.get('mismatched_layer_2_removed', 0) > 0:
                    print(f"     - Mismatched Layer_2 removed: {auto_fix_stats['mismatched_layer_2_removed']}")
                if auto_fix_stats.get('orphaned_layer_1_removed', 0) > 0:
                    print(f"     - Orphaned Layer_1 removed: {auto_fix_stats['orphaned_layer_1_removed']}")
                print(f"     - Total labels removed: {auto_fix_stats.get('total_labels_removed', 0)}")
            
            # Show breakdown of invalid reasons if any
            invalid_reasons = validation_stats.get('invalid_reasons', {})
            if invalid_reasons and any(count > 0 for count in invalid_reasons.values()):
                print(f"   • Invalid reasons breakdown:")
                for reason, count in invalid_reasons.items():
                    if count > 0:
                        reason_display = reason.replace('_', ' ').title()
                        print(f"     - {reason_display}: {count}")
            
            quarantine_dir = validation_stats.get('quarantine_directory')
            if quarantine_dir:
                print(f"   • Quarantine directory: {quarantine_dir}")
        
        print(f"\n💾 Output saved to: {config['output_dir']}")
        
        # Save detailed results to JSON
        results_file = os.path.join(config['output_dir'], 'preprocessing_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📝 Detailed results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1
    finally:
        # Ensure all progress bars are properly closed
        try:
            progress_callback.cleanup()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
