"""
Factory untuk membuat dan menampilkan modul UI Pretrained.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Pretrained menggunakan BaseUIModule dan UI Factory pattern.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/pretrained/pretrained_ui_factory.py
"""

from typing import Dict, Any, Optional, Type, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.pretrained.pretrained_uimodule import PretrainedUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class PretrainedUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Pretrained.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Pretrained dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_pretrained_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PretrainedUIModule:
        """
        Buat instance PretrainedUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Instance PretrainedUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug("Membuat instance PretrainedUIModule")
            
            # Create instance directly since PretrainedUIModule handles its own initialization
            module = PretrainedUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug("✅ Berhasil membuat instance PretrainedUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat PretrainedUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_pretrained(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Buat dan tampilkan modul Pretrained UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Dict berisi informasi modul atau None jika gagal
        """
        try:
            logger.debug("Membuat dan menampilkan Pretrained UI")
            
            # Buat instance modul
            module = cls.create_pretrained_module(config=config, **kwargs)
            
            # Tampilkan UI
            display_result = module.display_ui()
            
            if display_result and not display_result.get('success', False):
                error_msg = display_result.get('message', 'Gagal menampilkan UI')
                logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            logger.debug("✅ Berhasil membuat dan menampilkan Pretrained UI")
            return {'success': True, 'module': module}
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Pretrained UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'message': error_msg, 'error': str(e)}


# Fungsi utilitas untuk kemudahan penggunaan
def create_pretrained_display(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Callable:
    """
    Buat fungsi display untuk modul Pretrained.
    
    Args:
        config: Konfigurasi opsional untuk modul
        **kwargs: Argumen tambahan untuk inisialisasi modul
        
    Returns:
        Fungsi yang dapat dipanggil untuk menampilkan UI
    """
    return PretrainedUIFactory.create_display_function(
        module_class=PretrainedUIModule,
        module_name="pretrained",
        config=config,
        **kwargs
    )
