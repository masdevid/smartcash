"""
Factory untuk membuat dan menampilkan modul UI Evaluation.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Evaluation menggunakan BaseUIModule dan UI Factory pattern.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/evaluation/evaluation_ui_factory.py
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class EvaluationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Evaluation.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Evaluation dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_evaluation_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationUIModule:
        """
        Buat instance EvaluationUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Instance EvaluationUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug("Membuat instance EvaluationUIModule")
            
            # Create instance directly since EvaluationUIModule handles its own initialization
            module = EvaluationUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug("✅ Berhasil membuat instance EvaluationUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat EvaluationUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_evaluation(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Buat dan tampilkan modul Evaluation UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
            
        Returns:
            Dict berisi informasi modul atau error message
        """
        try:
            logger.debug("Membuat dan menampilkan Evaluation UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_evaluation_module(config=config, **kwargs)
            
            # Let the module handle its own display logic
            if auto_display:
                # Just get the main container without displaying it here
                # The module will handle display through its own display_ui method
                logger.debug("✅ Evaluation UI module created and ready for display")
            else:
                logger.debug("✅ Evaluation UI module created (auto-display disabled)")
            
            # Return the module itself to allow for more flexible usage
            # The caller can choose to display it or access components directly
            return module
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Evaluation UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'message': error_msg}


def create_evaluation_display(**kwargs) -> callable:
    """
    Create a display function for the evaluation UI.
    
    This is a convenience function that returns a callable that can be used
    to display the evaluation UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the evaluation UI
        
    Returns:
        A callable that will display the evaluation UI when called
    """
    def display_fn():
        return EvaluationUIFactory.create_and_display_evaluation(**kwargs)
    
    return display_fn
