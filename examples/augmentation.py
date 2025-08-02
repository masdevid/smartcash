#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/examples/augmentation.py

Optimized augmentation script with cleanup API support and minimal logging.
Uses AugmentationService with environment-aware configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import augmentation service
from smartcash.dataset.augmentor.service import AugmentationService
from smartcash.dataset.augmentor.utils import get_default_augmentation_config
from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
from smartcash.common.logger import get_logger

# Setup logger - only warnings, errors, and phase transitions
logger = get_logger('augmentation_example', level="WARNING")

def cleanup_augmented_data(output_dir: str, splits: List[str] = None) -> Dict[str, Any]:
    """Wrapper for CleanupManager API that targets both /data/augmented and /data/preprocessed"""
    try:
        # Create config for cleanup manager
        data_root = str(Path(output_dir).parent)
        config = {'data': {'dir': data_root}}
        cleanup_manager = CleanupManager(config)
        
        # If no splits specified, clean all standard splits
        if not splits:
            splits = ['train', 'valid', 'test']
        
        # Track cleanup results by split
        total_removed = 0
        split_results = {}
        
        for split in splits:
            # Use CleanupManager API which handles both augmented and preprocessed directories
            result = cleanup_manager.cleanup_augmented_data(target_split=split)
            split_removed = result.get('total_removed', 0)
            total_removed += split_removed
            
            split_results[split] = {
                'status': result.get('status', 'unknown'),
                'files_removed': split_removed,
                'message': result.get('message', '')
            }
        
        return {
            'success': total_removed >= 0,  # Consider success if no errors
            'message': f'Cleaned {total_removed} files across {len(splits)} splits (augmented + preprocessed)',
            'total_removed': total_removed,
            'by_split': split_results
        }
    except Exception as e:
        return {'success': False, 'message': str(e), 'total_removed': 0}

def create_augmentation_config(args) -> Dict[str, Any]:
    """Create augmentation configuration from command line arguments"""
    config = get_default_augmentation_config()
    
    # Update config with command line arguments
    if args.num_variations:
        config['augmentation']['num_variations'] = args.num_variations
    if args.target_count:
        config['augmentation']['target_count'] = args.target_count
    if args.augmentation_types:
        config['augmentation']['types'] = args.augmentation_types.split(',')
    if args.intensity:
        config['augmentation']['intensity'] = args.intensity
    
    # Set input/output directories
    config['paths'] = {
        'raw': str(Path(args.input_dir).resolve()),
        'augmented': str(Path(args.output_dir).resolve())
    }
    
    # Set normalization config
    config['preprocessing'] = {
        'normalization': {
            'type': args.normalization,
            'target_size': [args.img_size, args.img_size]
        }
    }
    
    return config

def create_optimized_progress_callback(verbose: bool = False) -> Callable:
    """Create an optimized progress callback with minimal logging."""
    progress_bars = {}
    current_phase = None
    
    def update_progress(phase: str, current: int, total: int, message: str = "", **kwargs):
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
    update_progress.cleanup = lambda: [bar.close() for bar in progress_bars.values()]
    return update_progress

def main():
    """Main function to run augmentation"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run dataset augmentation')
    parser.add_argument('--input-dir', type=str, default=os.getenv('DATA_DIR', 'data'),
                      help='Input directory containing dataset splits (default: data/)')
    parser.add_argument('--output-dir', type=str, default=os.getenv('AUGMENTED_DIR', 'data/augmented'),
                      help='Output directory for augmented data (default: data/augmented/)')
    parser.add_argument('--splits', type=str, default='train',
                      help='Comma-separated list of splits to augment (default: train)')
    parser.add_argument('--num-variations', type=int, default=int(os.getenv('AUG_NUM_VARIATIONS', 3)),
                      help='Number of augmented variations per image (default: 3)')
    parser.add_argument('--target-count', type=int, default=int(os.getenv('AUG_TARGET_COUNT', 1000)),
                      help='Target number of images per class (default: 1000)')
    parser.add_argument('--augmentation-types', type=str, default=os.getenv('AUG_TYPES', 'combined'),
                      help='Comma-separated list of augmentation types (default: combined)')
    parser.add_argument('--intensity', type=float, default=float(os.getenv('AUG_INTENSITY', '0.8')),
                      help='Augmentation intensity level (0.0-1.0, default: 0.8)')
    parser.add_argument('--normalization', type=str, default=os.getenv('NORMALIZATION', 'default'),
                      help='Normalization preset (default: default)')
    parser.add_argument('--img-size', type=int, default=int(os.getenv('IMG_SIZE', 640)),
                      help='Image size for preprocessing (default: 640)')
    parser.add_argument('--workers', type=int, default=int(os.getenv('NUM_WORKERS', 8)),
                      help='Number of worker processes (default: 8)')
    parser.add_argument('--cleanup', action='store_true',
                      help='Clean up existing augmented data before starting')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging (shows all messages)')
    
    args = parser.parse_args()
    
    # Configure logger based on verbose flag
    global logger
    if args.verbose:
        logger = get_logger('augmentation_example', level="INFO")
    
    # Convert comma-separated string to list
    splits = [s.strip() for s in args.splits.split(',')]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up existing augmented data if requested
    if args.cleanup:
        print("🧹 Cleaning up existing augmented data...")
        try:
            cleanup_result = cleanup_augmented_data(args.output_dir, splits)
            if cleanup_result.get('success'):
                print("✅ Cleanup completed successfully")
                if args.verbose and 'by_split' in cleanup_result:
                    for split, stats in cleanup_result['by_split'].items():
                        files_removed = stats.get('files_removed', 0)
                        status = stats.get('status', 'unknown')
                        print(f"   • {split}: {files_removed} files removed (status: {status})")
            else:
                logger.warning(f"⚠️ Cleanup warning: {cleanup_result.get('message', 'Unknown issue')}")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 1
    
    # Display configuration
    print("=" * 60)
    print("🚀 SmartCash Dataset Augmentation")
    print("=" * 60)
    print(f"📂 Input directory: {args.input_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🔢 Splits: {', '.join(splits)}")
    print(f"🔄 Augmentation types: {args.augmentation_types}")
    print(f"✨ Variations per image: {args.num_variations}")
    print(f"🎯 Target count per class: {args.target_count}")
    print(f"⚡ Intensity level: {args.intensity}")
    print(f"🖼️  Image size: {args.img_size}x{args.img_size}")
    print(f"🔧 Normalization: {args.normalization}")
    print(f"⚡ Workers: {args.workers}")
    print(f"🧹 Cleanup mode: {'Enabled' if args.cleanup else 'Disabled'}")
    print(f"📝 Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print("-" * 60)
    
    try:
        # Create configuration
        config = create_augmentation_config(args)
        
        # Create optimized progress callback
        progress_callback = create_optimized_progress_callback(args.verbose)
        
        # Update config with number of workers
        if args.workers > 1:
            config['augmentation']['num_workers'] = args.workers
            if args.verbose:
                print(f"⚙️  Using {args.workers} worker(s) for augmentation")
        
        # Initialize augmentation service
        service = AugmentationService(config, progress_callback)
        
        # Initialize results dictionary to store results for each split
        all_results = {}
        
        try:
            # Run augmentation for each split
            for split in splits:
                print(f"🚀 Starting augmentation pipeline for {split} split...")
                
                # Run augmentation pipeline
                results = service.run_augmentation_pipeline(split)
                all_results[split] = results
                
                # Display results with summary
                print(f"\n✅ Augmentation completed for {split} split:")
                print(f"   • Generated images: {results.get('total_generated', 0)}")
                print(f"   • Normalized images: {results.get('total_normalized', 0)}")
                
                # Save detailed results to JSON
                split_output_dir = os.path.join(args.output_dir, split)
                os.makedirs(split_output_dir, exist_ok=True)
                results_file = os.path.join(split_output_dir, 'augmentation_results.json')
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                if args.verbose:
                    print(f"   • Results saved to: {results_file}")
                
        finally:
            # Ensure any progress bars are properly closed
            try:
                progress_callback.cleanup()
            except:
                pass
        
        print("\n" + "=" * 60)
        print("🎉 Augmentation completed successfully!")
        print(f"💾 Output saved to: {args.output_dir}")
        
        # Show overall statistics
        print("\n📊 Overall Statistics:")
        total_generated = sum(r.get('total_generated', 0) for r in all_results.values())
        total_normalized = sum(r.get('total_normalized', 0) for r in all_results.values())
        print(f"   • Total generated images: {total_generated}")
        print(f"   • Total normalized images: {total_normalized}")
        print(f"   • Splits processed: {len(all_results)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Augmentation failed: {str(e)}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1
    finally:
        # Ensure progress callback cleanup
        try:
            progress_callback.cleanup()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
