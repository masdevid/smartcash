"""
File: smartcash/ui/dataset/augmentation/operations/augment_cleanup_operation.py
Description: Cleanup operation for removing temporary files and artifacts after augmentation.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .augmentation_base_operation import AugmentationBaseOperation, OperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule

class AugmentCleanupOperation(AugmentationBaseOperation):
    """
    Cleanup operation that removes temporary files and artifacts after augmentation.
    
    This class handles the cleanup of temporary directories, cached files, and other
    artifacts created during the augmentation process.
    """
    
    def __init__(
        self,
        ui_module: 'AugmentationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the cleanup operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for cleanup
            callbacks: Optional callbacks for operation events
        """
        super().__init__(ui_module, config, callbacks)
        self._removed_paths = []
    
    def _cleanup_temp_files(self, results: Dict[str, Any]) -> None:
        """Clean up temporary files and directories."""
        temp_dirs = [
            self._config.get('temp_dir'),
            self._config.get('cache_dir'),
            os.path.join(os.getcwd(), '.temp')
        ]
        
        for temp_dir in filter(None, temp_dirs):
            if not os.path.exists(temp_dir):
                continue
                
            try:
                self.log_info(f"Menghapus direktori sementara: {temp_dir}")
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
                    self._removed_paths.append(temp_dir)
                    results['removed'].append(temp_dir)
            except Exception as e:
                error_msg = f"Gagal menghapus {temp_dir}: {str(e)}"
                self.log_error(error_msg)
                results['errors'].append(error_msg)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the cleanup operation.
        
        Returns:
            Dictionary containing cleanup results
        """
        self.log_operation_start("Cleaning Up")
        self.log('Memulai pembersihan...', 'info')
        
        try:
            # Get cleanup configuration
            cleanup_config = self._config.get('cleanup', {})
            if not cleanup_config:
                return self._handle_error("Tidak ada konfigurasi cleanup yang ditemukan")
            
            # Get backend API if needed
            cleanup_backend = self.get_backend_api('cleanup')
            
            # Determine what to clean up
            target = cleanup_config.get('target', 'temp')
            self.log_info(f"Membersihkan target: {target}")
            
            results = {
                'status': 'success',
                'removed': [],
                'errors': []
            }
            
            # Handle different cleanup targets
            if target == 'temp' or target == 'both':
                self._cleanup_temp_files(results)
            
            if (target == 'output' or target == 'both') and 'output_dir' in self._config:
                self._cleanup_output_dir(self._config['output_dir'], results)
            
            if cleanup_backend and 'backend' in cleanup_config.get('targets', []):
                self._cleanup_backend(cleanup_backend, results)
            
            # Update status
            if results['errors']:
                self.log(f"⚠️ Pembersihan selesai dengan {len(results['errors'])} kesalahan", 'warning')
            else:
                self.log('✅ Pembersihan selesai', 'info')
            
            results['status'] = 'success' if not results['errors'] else 'warning'
            return results
            
        except Exception as e:
            error_msg = f"Terjadi kesalahan saat membersihkan: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg)
            
        finally:
            self.log_operation_complete("Cleanup")
    
    def _cleanup_temp_files(self, results: Dict[str, Any]) -> None:
        """Clean up temporary files and directories."""
        temp_dirs = [
            self._config.get('temp_dir'),
            self._config.get('cache_dir'),
            os.path.join(os.getcwd(), '.temp')
        ]
        
        for temp_dir in filter(None, temp_dirs):
            if not os.path.exists(temp_dir):
                continue
                
            try:
                self.log_info(f"Menghapus direktori sementara: {temp_dir}")
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
                    self._removed_paths.append(temp_dir)
                    results['removed'].append(temp_dir)
            except Exception as e:
                error_msg = f"Gagal menghapus {temp_dir}: {str(e)}"
                self.log_error(error_msg)
                results['errors'].append(error_msg)
    
    def _cleanup_output_dir(self, output_dir: str, results: Dict[str, Any]) -> None:
        """Clean up output directory based on configuration."""
        if not os.path.exists(output_dir):
            return
            
        try:
            cleanup_config = self._config.get('cleanup', {})
            
            # Check if we should keep any files
            if cleanup_config.get('keep_original', True):
                self.log_info("Menyimpan file asli, hanya membersihkan file hasil augmentasi")
                # Implement logic to keep original files and remove only augmented ones
                # This would depend on how your augmented files are named/stored
                pass
            else:
                self.log_info(f"Menghapus seluruh direktori output: {output_dir}")
                shutil.rmtree(output_dir)
                self._removed_paths.append(output_dir)
                results['removed'].append(output_dir)
                
        except Exception as e:
            error_msg = f"Gagal membersihkan direktori output {output_dir}: {str(e)}"
            self.log_error(error_msg)
            results['errors'].append(error_msg)
    
    def _cleanup_backend(self, backend_api: Callable, results: Dict[str, Any]) -> None:
        """Perform backend-specific cleanup."""
        try:
            self.log_info("Menjalankan pembersihan backend...")
            result = backend_api(self._config)
            
            if result.get('status') == 'success':
                removed = result.get('removed', [])
                self._removed_paths.extend(removed)
                results['removed'].extend(removed)
                self.log_info(f"Backend menghapus {len(removed)} item")
            else:
                error_msg = result.get('message', 'Gagal membersihkan backend')
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Gagal membersihkan backend: {str(e)}"
            self.log_error(error_msg)
            results['errors'].append(error_msg)
    
    def get_removed_paths(self) -> list:
        """Get list of paths that were removed during cleanup."""
        return self._removed_paths


# Factory function has been moved to augment_factory.py
