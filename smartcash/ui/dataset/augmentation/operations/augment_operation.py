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

    def _progress_adapter(self, progress: float, message: str = '', level: str = 'overall', secondary_progress: Optional[float] = None, secondary_message: str = '') -> None:
        """
        Adapter for progress callbacks from the backend with dual progress support.
        
        Args:
            progress: Progress value between 0 and 1 for main progress
            message: Optional progress message for main progress
            level: Progress level ('overall' or 'current')
            secondary_progress: Optional secondary progress value between 0 and 1
            secondary_message: Optional secondary progress message
        """
        # Convert progress to percentage
        main_progress_pct = int(progress * 100)
        
        # Handle dual progress if secondary_progress is provided
        if secondary_progress is not None:
            secondary_progress_pct = int(secondary_progress * 100)
            self.update_progress(
                progress=main_progress_pct,
                message=message,
                secondary_progress=secondary_progress_pct,
                secondary_message=secondary_message
            )
        else:
            # Single progress mode (backward compatibility)
            self.update_progress(main_progress_pct, message)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the augmentation operation.
        
        Returns:
            Dictionary containing operation results
        """
        self.log_operation_start("Augmenting Dataset")
        self.log('Memulai proses augmentasi...', 'info')
        
        try:
            # Get backend API
            run_pipeline = self.get_backend_api('run_pipeline')
            if not run_pipeline:
                return self._handle_error("Backend augmentation service not available")
            
            # Execute augmentation pipeline
            self.log('Menjalankan pipeline augmentasi...', 'info')
            
            try:
                result = run_pipeline(
                    config=self._config,
                    progress_callback=self._progress_adapter
                )
                
                if result.get('status') != 'success':
                    error_msg = result.get('message', 'Gagal menjalankan augmentasi')
                    return self._handle_error(error_msg, result.get('error'))
                
                # Update operation status
                self.log('✅ Augmentasi selesai', 'info')
                
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
