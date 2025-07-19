"""
Factory untuk membuat dan menampilkan modul UI Training.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Training menggunakan BaseUIModule dan UI Factory pattern.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/training/training_ui_factory.py
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.training.training_uimodule import TrainingUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class TrainingUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Training.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Training dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_training_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TrainingUIModule:
        """
        Buat instance TrainingUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Instance TrainingUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug("Membuat instance TrainingUIModule")
            
            # Create instance directly since TrainingUIModule handles its own initialization
            module = TrainingUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug("✅ Berhasil membuat instance TrainingUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat TrainingUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_training(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Buat dan tampilkan modul Training UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
            
        Returns:
            Dict berisi informasi modul atau None jika gagal
        """
        try:
            logger.debug("Membuat dan menampilkan Training UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_training_module(config=config, **kwargs)
            
            # Let the module handle its own display logic
            if auto_display:
                # Just get the main container without displaying it here
                # The module will handle display through its own display_ui method
                logger.debug("✅ Training UI module created and ready for display")
            else:
                logger.debug("✅ Training UI module created (auto-display disabled)")
            
            # Return the module itself to allow for more flexible usage
            # The caller can choose to display it or access components directly
            return module
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Training UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'message': error_msg}


# Fungsi utilitas untuk kemudahan penggunaan
def create_training_display(**kwargs) -> Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]:
    """
    Buat fungsi display untuk modul Training UI.
    
    Contoh penggunaan:
        from smartcash.ui.model.training import create_training_display
        show_training = create_training_display()
        show_training(config=my_config)
    
    Args:
        **kwargs: Argumen tambahan untuk TrainingUIFactory.create_training_display
        
    Returns:
        Fungsi yang dapat dipanggil untuk menampilkan modul Training UI
    """
    return TrainingUIFactory.create_training_display(**kwargs)
