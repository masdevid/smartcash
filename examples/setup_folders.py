"""
setup_folders.py

A simplified version of the folders operation without UI components.
This script creates required directories in parallel with error handling and progress reporting.
"""

import os
import concurrent.futures
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Define required folders (relative paths for local development)
REQUIRED_FOLDERS = [
    "data",
    "data/downloads",
    "data/backup",
    "data/train",
    "data/test",
    "data/valid",
    "data/invalid",
    "data/preprocessed",
    "data/augmented",
    "data/pretrained",
    "models",
    "configs",
    "outputs",
    "logs"
]


def log_message(level: str, message: str) -> None:
    """Simple logging function."""
    print(f"[{level.upper()}] {message}")


def create_single_folder(folder_path: str) -> Dict[str, Any]:
    """
    Create a single folder with error handling.
    
    Args:
        folder_path: Path to the folder to create
        
    Returns:
        Dict containing operation result
    """
    try:
        path_obj = Path(folder_path)
        
        # Create directory with parents, no error if exists
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Verify the directory was created
        if path_obj.exists() and path_obj.is_dir():
            log_message("info", f"Created directory: {folder_path}")
            return {
                'path': folder_path,
                'name': path_obj.name,
                'status': 'created',
                'success': True
            }
        else:
            error_msg = f"Failed to verify directory: {folder_path}"
            log_message("error", error_msg)
            return {
                'path': folder_path,
                'name': path_obj.name,
                'success': False,
                'error': error_msg
            }
            
    except Exception as e:
        error_msg = f"Error creating {folder_path}: {str(e)}"
        log_message("error", error_msg)
        return {
            'path': folder_path,
            'name': os.path.basename(folder_path),
            'success': False,
            'error': str(e)
        }


def create_folders_parallel(folders: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """
    Create multiple folders in parallel using ThreadPoolExecutor.
    
    Args:
        folders: List of folder paths to create
        max_workers: Maximum number of worker threads
        
    Returns:
        Dict with creation results
    """
    created_folders = []
    failed_folders = []
    
    # Filter out existing folders
    folders_to_create = []
    for folder_path in folders:
        if not os.path.exists(folder_path):
            folders_to_create.append(folder_path)
        else:
            log_message("info", f"Directory already exists: {folder_path}")
            created_folders.append({
                'path': folder_path,
                'status': 'exists',
                'name': os.path.basename(folder_path)
            })
    
    if not folders_to_create:
        log_message("info", "All folders already exist - nothing to create")
        return {
            'successful': len(folders),
            'failed': 0,
            'skipped': len(folders)
        }
    
    # Create folders in parallel
    successful_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all folder creation tasks
        future_to_folder = {
            executor.submit(create_single_folder, folder_path): folder_path 
            for folder_path in folders_to_create
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_folder):
            folder_path = future_to_folder[future]
            try:
                result = future.result(timeout=5)  # 5 second timeout per folder
                if result['success']:
                    successful_count += 1
                    created_folders.append(result)
                else:
                    failed_count += 1
                    failed_folders.append(result)
                    
            except concurrent.futures.TimeoutError:
                failed_count += 1
                error_result = {
                    'path': folder_path,
                    'success': False,
                    'error': 'Creation timeout'
                }
                failed_folders.append(error_result)
                log_message("error", f"Timeout creating folder: {folder_path}")
                
            except Exception as e:
                failed_count += 1
                error_result = {
                    'path': folder_path,
                    'success': False,
                    'error': str(e)
                }
                failed_folders.append(error_result)
                log_message("error", f"Exception creating {folder_path}: {e}")
    
    # Calculate summary
    total_successful = successful_count + len([f for f in created_folders if f.get('status') == 'exists'])
    
    log_message("info", f"Folder creation completed: {total_successful} successful, {failed_count} failed")
    
    return {
        'successful': total_successful,
        'failed': failed_count,
        'created_new': successful_count,
        'already_existed': len([f for f in created_folders if f.get('status') == 'exists']),
        'created_folders': created_folders,
        'failed_folders': failed_folders
    }


def verify_folders(folders: List[str]) -> Dict[str, Any]:
    """
    Verify that the required folders exist.
    
    Args:
        folders: List of folder paths to verify
        
    Returns:
        Dict with verification results
    """
    missing_folders = []
    existing_folders = []
    
    for folder_path in folders:
        path_obj = Path(folder_path)
        if path_obj.exists() and path_obj.is_dir():
            existing_folders.append(folder_path)
        else:
            missing_folders.append(folder_path)
    
    success_rate = (len(existing_folders) / len(folders)) * 100 if folders else 0
    
    return {
        'success': len(missing_folders) == 0,
        'existing_folders': existing_folders,
        'missing_folders': missing_folders,
        'total_folders': len(folders),
        'success_rate': success_rate
    }


def main():
    """Main function to execute folder creation and verification."""
    print("🚀 Starting folder setup...")
    start_time = time.time()
    
    # Create folders in parallel
    result = create_folders_parallel(REQUIRED_FOLDERS)
    
    # Verify the folders
    verification = verify_folders(REQUIRED_FOLDERS)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n📊 Summary:")
    print(f"  • Total folders: {result['successful'] + result['failed']}")
    print(f"  • Created: {result['created_new']}")
    print(f"  • Already existed: {result['already_existed']}")
    print(f"  • Failed: {result['failed']}")
    print(f"  • Success rate: {verification['success_rate']:.1f}%")
    print(f"  • Execution time: {execution_time:.2f} seconds")
    
    if verification['missing_folders']:
        print("\n❌ Missing folders:")
        for folder in verification['missing_folders']:
            print(f"  - {folder}")
    
    if result['failed'] > 0:
        print("\n❌ Failed to create folders:")
        for folder in result.get('failed_folders', []):
            print(f"  - {folder['path']}: {folder.get('error', 'Unknown error')}")
    
    if verification['success']:
        print("\n✅ All folders are set up successfully!")
    else:
        print(f"\n⚠️  Some folders could not be created. Success rate: {verification['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
