#!/usr/bin/env python3
"""
Example script demonstrating how to use the SmartCash dataset downloader API.

This script shows how to:
1. Initialize a download session
2. Configure download parameters
3. Track download progress with minimal logging
4. Use cleanup API properly
5. Handle errors gracefully

Usage:
    python download.py --api-key YOUR_ROBOFLOW_API_KEY [--workspace WORKSPACE] [--project PROJECT] [--version VERSION]
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path to ensure imports work
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.dataset.downloader.api.downloader_api import (
    create_download_session,
    get_default_config,
    validate_downloader_config,
    get_cleanup_service
)
from smartcash.common.logger import get_logger

# Configure logging - only warnings, errors, and phase transitions
logger = get_logger("download_example", level="WARNING")

def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='Download dataset from Roboflow')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Roboflow API key (default: load from .env)')
    parser.add_argument('--workspace', type=str, default='smartcash-wo2us',
                       help='Roboflow workspace name (default: %(default)s)')
    parser.add_argument('--project', type=str, default='rupiah-emisi-2022',
                       help='Project name (default: %(default)s)')
    parser.add_argument('--version', type=str, default='3',
                       help='Dataset version (default: %(default)s)')
    parser.add_argument('--output-dir', type=str, default='./data/downloads',
                       help='Output directory for downloaded dataset (default: %(default)s)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up existing downloads before starting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (shows all messages)')
    return parser

def cleanup_downloads(output_dir: str) -> Dict[str, Any]:
    """Simple wrapper for cleanup functionality"""
    try:
        cleanup_service = get_cleanup_service()
        return cleanup_service.cleanup_downloads_only(Path(output_dir))
    except Exception as e:
        return {'success': False, 'message': str(e)}

def download_dataset(args) -> bool:
    """
    Download dataset using the provided arguments with optimized progress tracking.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up existing downloads if requested
    if args.cleanup:
        print("🧹 Cleaning up existing downloads...")
        try:
            cleanup_result = cleanup_downloads(args.output_dir)
            if cleanup_result.get('success'):
                print("✅ Cleanup completed successfully")
            else:
                logger.warning(f"⚠️ Cleanup warning: {cleanup_result.get('message', 'Unknown issue')}")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    # Create download configuration
    config = get_default_config(api_key=args.api_key)
    config.update({
        'workspace': args.workspace,
        'project': args.project,
        'version': args.version,
        'output_dir': args.output_dir,
    })
    
    # Validate configuration
    if not validate_downloader_config(config):
        logger.error("❌ Invalid download configuration")
        return False
    
    # Create download session
    print("🚀 Initializing download session...")
    session = create_download_session(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=args.output_dir,
        logger=logger
    )
    
    if not session or not session.get('ready', False):
        logger.error("❌ Failed to create download session")
        return False
    
    download_service = session['service']
    
    # Set up optimized progress callback with tqdm
    if session.get('has_progress_callback', False):
        # Phase mapping for better progress tracking
        phase_map = {
            'init': (0, 10, "🚀 Initializing"),
            'metadata': (10, 20, "📋 Fetching metadata"),
            'backup': (20, 25, "💾 Creating backup"),
            'download': (25, 70, "⬇️ Downloading files"),
            'extract': (70, 85, "📦 Extracting archive"),
            'organize': (85, 95, "🗂️ Organizing files"),
            'uuid_rename': (95, 98, "🏷️ Renaming files"),
            'validate': (98, 99, "✅ Validating"),
            'cleanup': (99, 100, "🧹 Cleanup"),
            'complete': (100, 100, "🎉 Complete")
        }
        
        progress_bar = tqdm(
            total=100,
            desc="🚀 Initializing",
            bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]{postfix}",
            postfix=""
        )
        
        last_progress = 0
        current_phase = None
        
        def progress_callback(step: str, current: int, total: int = 100, message: str = ""):
            """Optimized progress callback with phase transition logging."""
            nonlocal last_progress, current_phase
            
            # Handle backend progress updates
            if isinstance(step, str):
                step_name = step.lower().strip()
                
                # Log phase transitions only
                if step_name != current_phase:
                    current_phase = step_name
                    if step_name in phase_map:
                        _, _, phase_desc = phase_map[step_name]
                        if args.verbose:
                            print(f"\n📍 Phase transition: {phase_desc}")
                
                # Calculate progress based on phase
                if step_name in phase_map:
                    phase_min, phase_max, phase_desc = phase_map[step_name]
                    # Calculate progress within phase
                    phase_progress = (current / total) * (phase_max - phase_min)
                    total_progress = phase_min + phase_progress
                else:
                    total_progress = current
            else:
                total_progress = step if isinstance(step, (int, float)) else current
            
            # Update progress bar
            total_progress = max(0, min(100, total_progress))
            progress_diff = total_progress - last_progress
            
            if progress_diff > 0:
                progress_bar.update(progress_diff)
                last_progress = total_progress
            
            # Update description and postfix
            if current_phase in phase_map:
                _, _, phase_desc = phase_map[current_phase]
                progress_bar.set_description(phase_desc)
            
            if message and args.verbose:
                progress_bar.set_postfix_str(message[:30], refresh=True)
            
        download_service.set_progress_callback(progress_callback)
    
    try:
        # Start download
        print(f"⬇️ Downloading dataset {args.workspace}/{args.project}/{args.version}...")
        result = download_service.download_dataset()
        success = result.get('status') == 'success'
        
        if success:
            if 'progress_bar' in locals():
                progress_bar.close()
            print(f"\n✅ Download completed successfully!")
            print(f"📁 Dataset saved to: {os.path.abspath(args.output_dir)}")
            
            # Log summary information
            if result.get('summary'):
                summary = result['summary']
                print("📊 Download Summary:")
                print(f"   • Total files: {summary.get('total_files', 'N/A')}")
                print(f"   • Total size: {summary.get('total_size', 'N/A')}")
                if summary.get('splits'):
                    for split, count in summary['splits'].items():
                        print(f"   • {split}: {count} files")
            
            return True
        else:
            error_msg = result.get('message', 'Unknown error')
            logger.error(f"❌ Download failed: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during download: {str(e)}")
        return False
    finally:
        # Cleanup resources properly
        try:
            if hasattr(download_service, 'cleanup'):
                download_service.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
        
        # Close progress bar if it exists
        if 'progress_bar' in locals():
            try:
                progress_bar.close()
            except:
                pass

def load_environment():
    """Load environment variables from .env file."""
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        pass  # Environment loaded successfully
    else:
        logger.warning(f"⚠️ .env file not found at {env_path}")
    
    return {
        'ROBOFLOW_API_KEY': os.getenv('ROBOFLOW_API_KEY'),
        'ROBOFLOW_WORKSPACE': os.getenv('ROBOFLOW_WORKSPACE', 'smartcash-wo2us'),
        'ROBOFLOW_PROJECT': os.getenv('ROBOFLOW_PROJECT', 'rupiah-emisi-2022'),
        'ROBOFLOW_VERSION': os.getenv('ROBOFLOW_VERSION', '3')
    }

def main():
    """Main function to run the download example."""
    # Load environment variables first
    env = load_environment()
    
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Configure logger based on verbose flag
    global logger
    if args.verbose:
        logger = get_logger("download_example", level="INFO")
    
    # Use environment variables as defaults if not provided via command line
    if not args.api_key and env['ROBOFLOW_API_KEY']:
        args.api_key = env['ROBOFLOW_API_KEY']
    if not args.workspace and env['ROBOFLOW_WORKSPACE']:
        args.workspace = env['ROBOFLOW_WORKSPACE']
    if not args.project and env['ROBOFLOW_PROJECT']:
        args.project = env['ROBOFLOW_PROJECT']
    if not args.version and env['ROBOFLOW_VERSION']:
        args.version = env['ROBOFLOW_VERSION']
    
    # Validate API key
    if not args.api_key:
        logger.error("❌ Roboflow API key is required. Either provide it via --api-key "
                   "or set ROBOFLOW_API_KEY in .env file")
        return 1
    
    print("=" * 60)
    print("📡 SmartCash Dataset Download Example")
    print("=" * 60)
    print(f"Workspace: {args.workspace}")
    print(f"Project: {args.project}")
    print(f"Version: {args.version}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Cleanup mode: {'Enabled' if args.cleanup else 'Disabled'}")
    print(f"Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print("-" * 60)
    
    success = download_dataset(args)
    
    if success:
        print("✨ All done!")
        return 0
    else:
        print("❌ Download failed")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("❌ Unexpected error occurred")
        sys.exit(1)
