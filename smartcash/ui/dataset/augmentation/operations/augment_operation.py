"""
File: smartcash/ui/dataset/augmentation/operations/augment_operation.py
Description: Core augmentation operation implementation following the augmentation service pattern.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .augmentation_base_operation import AugmentationBaseOperation, OperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule

class AugmentOperation(AugmentationBaseOperation):
    """
    Core augmentation operation that processes images with specified augmentations.
    
    This class handles the execution of image augmentation with progress tracking,
    error handling, and UI integration.
    """
    
    def __init__(
        self,
        ui_module: 'AugmentationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the augmentation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for augmentation
            callbacks: Optional callbacks for operation events
        """
        super().__init__(ui_module, config, callbacks)
        self._processed_count = 0
        self._total_to_process = 0

    def _progress_adapter(self, progress: float, message: str = '') -> None:
        """
        Adapter for progress callbacks from the backend.
        
        Args:
            progress: Progress value between 0 and 1
            message: Optional progress message
        """
        self.update_progress(progress * 100, message)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the augmentation operation.
        
        Returns:
            Dictionary containing operation results
        """
        self.log_operation_start("Augmenting Dataset")
        self.update_operation_status('Memulai proses augmentasi...', 'info')
        
        try:
            # Get backend API
            run_pipeline = self.get_backend_api('run_pipeline')
            if not run_pipeline:
                return self._handle_error("Backend augmentation service not available")
            
            # Execute augmentation pipeline
            self.update_operation_status('Menjalankan pipeline augmentasi...', 'info')
            
            try:
                result = run_pipeline(
                    config=self._config,
                    progress_callback=self._progress_adapter
                )
                
                if result.get('status') != 'success':
                    error_msg = result.get('message', 'Gagal menjalankan augmentasi')
                    return self._handle_error(error_msg, result.get('error'))
                
                # Update operation status
                self.update_operation_status('Augmentasi selesai', 'success')
                
                return {
                    'status': 'success',
                    'message': 'Augmentasi dataset berhasil',
                    'output_dir': result.get('output_dir'),
                    'processed_count': result.get('processed_count', 0),
                    'generated_count': result.get('generated_count', 0)
                }
                
            except Exception as e:
                error_msg = f"Kesalahan saat menjalankan pipeline: {str(e)}"
                return self._handle_error(error_msg, e)
            
        except Exception as e:
            error_msg = f"Terjadi kesalahan saat augmentasi: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg)
            
        finally:
            self.log_operation_complete("Augment Dataset")
