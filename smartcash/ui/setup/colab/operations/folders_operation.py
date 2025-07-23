"""
Folders Operation (Optimized) - Fast & Efficient Implementation
Create required folders in Colab with optimized performance.
"""

import os
import concurrent.futures
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from smartcash.ui.components.operation_container import OperationContainer
from ..constants import REQUIRED_FOLDERS
from .base_colab_operation import BaseColabOperation


class FoldersOperation(BaseColabOperation):
    """Optimized folders operation with parallel processing and minimal overhead."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        super().__init__(operation_name, config, operation_container, **kwargs)
        self._created_folders = []
        self._failed_folders = []
    
    def get_operations(self) -> Dict[str, Callable]:
        return {'create_folders': self.execute_create_folders}
    
    def execute_create_folders(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create required folders with optimized parallel processing."""
        def execute_operation():
            # Step 1: Initialize (5%)
            self.update_progress_safe(progress_callback, 5, "🔧 Initializing folder creation...", 10)
            
            # Step 2: Fast batch folder creation (90%)
            self.update_progress_safe(progress_callback, 10, "📁 Creating folders in parallel...", 20)
            
            start_time = self._get_current_time()
            creation_result = self._create_folders_parallel()
            end_time = self._get_current_time()
            
            # Step 3: Quick verification (5%)
            self.update_progress_safe(progress_callback, 95, "✅ Verifying folder structure...", 95)
            verification_result = self._quick_verify_folders()
            
            execution_time = end_time - start_time
            
            return self.create_success_result(
                f'Created {creation_result["successful"]}/{len(REQUIRED_FOLDERS)} folders in {execution_time:.2f}s',
                successful_folders=creation_result["successful"],
                failed_folders=creation_result["failed"],
                execution_time=execution_time,
                verification=verification_result,
                total_folders=len(REQUIRED_FOLDERS)
            )
        
        return self.execute_with_error_handling(execute_operation)
    
    def _create_folders_parallel(self) -> Dict[str, Any]:
        """Create folders using parallel processing for maximum speed."""
        try:
            # Prepare folder data efficiently
            folders_to_create = []
            for folder_path in REQUIRED_FOLDERS:
                if not os.path.exists(folder_path):
                    folders_to_create.append(folder_path)
                else:
                    self._created_folders.append({
                        'path': folder_path,
                        'status': 'exists',
                        'name': os.path.basename(folder_path)
                    })
            
            if not folders_to_create:
                self.log_info("All folders already exist - skipping creation")
                return {
                    'successful': len(REQUIRED_FOLDERS),
                    'failed': 0,
                    'skipped': len(REQUIRED_FOLDERS)
                }
            
            # Parallel folder creation with ThreadPoolExecutor
            successful_count = 0
            failed_count = 0
            
            # Optimize thread count based on number of folders
            max_workers = min(len(folders_to_create), 8)  # Cap at 8 threads
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all folder creation tasks
                future_to_folder = {
                    executor.submit(self._create_single_folder_fast, folder_path): folder_path 
                    for folder_path in folders_to_create
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_folder):
                    folder_path = future_to_folder[future]
                    try:
                        result = future.result(timeout=5)  # 5 second timeout per folder
                        if result['success']:
                            successful_count += 1
                            self._created_folders.append(result)
                            self.log_debug(f"✅ Created: {os.path.basename(folder_path)}")
                        else:
                            failed_count += 1
                            self._failed_folders.append(result)
                            self.log_error(f"❌ Failed: {os.path.basename(folder_path)} - {result.get('error', 'Unknown error')}")
                            
                    except concurrent.futures.TimeoutError:
                        failed_count += 1
                        self._failed_folders.append({
                            'path': folder_path,
                            'success': False,
                            'error': 'Creation timeout'
                        })
                        self.log_error(f"❌ Timeout: {os.path.basename(folder_path)}")
                    except Exception as e:
                        failed_count += 1
                        self._failed_folders.append({
                            'path': folder_path,
                            'success': False,
                            'error': str(e)
                        })
                        self.log_error(f"❌ Exception: {os.path.basename(folder_path)} - {e}")
            
            # Add existing folders to successful count
            total_successful = successful_count + len([f for f in self._created_folders if f['status'] == 'exists'])
            
            self.log_info(f"Folder creation completed: {total_successful} successful, {failed_count} failed")
            
            return {
                'successful': total_successful,
                'failed': failed_count,
                'created_new': successful_count,
                'already_existed': len([f for f in self._created_folders if f['status'] == 'exists'])
            }
            
        except Exception as e:
            self.log_error(f"Parallel folder creation failed: {e}")
            return {'successful': 0, 'failed': len(REQUIRED_FOLDERS), 'error': str(e)}
    
    def _create_single_folder_fast(self, folder_path: str) -> Dict[str, Any]:
        """Create a single folder with minimal overhead."""
        try:
            # Use pathlib for better performance and error handling
            path_obj = Path(folder_path)
            
            # Create directory with parents, no error if exists
            path_obj.mkdir(parents=True, exist_ok=True)
            
            # Quick verification - just check if it exists and is directory
            if path_obj.exists() and path_obj.is_dir():
                return {
                    'path': folder_path,
                    'name': path_obj.name,
                    'status': 'created',
                    'success': True
                }
            else:
                return {
                    'path': folder_path,
                    'name': path_obj.name,
                    'success': False,
                    'error': 'Creation verification failed'
                }
                
        except Exception as e:
            return {
                'path': folder_path,
                'name': os.path.basename(folder_path),
                'success': False,
                'error': str(e)
            }
    
    def _quick_verify_folders(self) -> Dict[str, Any]:
        """Quick verification of essential folders only."""
        try:
            # Quick batch verification using pathlib
            missing_folders = []
            existing_folders = []
            
            for folder_path in REQUIRED_FOLDERS:
                path_obj = Path(folder_path)
                if path_obj.exists() and path_obj.is_dir():
                    existing_folders.append(folder_path)
                else:
                    missing_folders.append(folder_path)
            
            success_rate = (len(existing_folders) / len(REQUIRED_FOLDERS)) * 100
            
            return {
                'success': len(missing_folders) == 0,
                'existing_folders': len(existing_folders),
                'missing_folders': len(missing_folders),
                'total_folders': len(REQUIRED_FOLDERS),
                'success_rate': success_rate,
                'critical_missing': missing_folders if missing_folders else []
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def get_progress_steps(self, operation_type: str = 'folders') -> list:
        """Get optimized progress steps for fast folder creation."""
        return [
            {'progress': 5, 'message': '🔧 Initializing...', 'phase_progress': 10},
            {'progress': 10, 'message': '📁 Creating folders...', 'phase_progress': 20},
            {'progress': 95, 'message': '✅ Verifying...', 'phase_progress': 95},
            {'progress': 100, 'message': '✅ Folders ready', 'phase_progress': 100}
        ]
    
    def get_folder_creation_status(self) -> Dict[str, Any]:
        """Get current status of folder creation operation."""
        return {
            'created_folders': len(self._created_folders),
            'failed_folders': len(self._failed_folders),
            'total_required': len(REQUIRED_FOLDERS),
            'success_rate': (len(self._created_folders) / len(REQUIRED_FOLDERS) * 100) if REQUIRED_FOLDERS else 0
        }