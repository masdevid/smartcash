"""
file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/core/ui_utils.py

Modul utilitas untuk operasi UI yang umum digunakan di seluruh aplikasi.
"""
import logging
from typing import Any, Dict, Optional, TypeVar, Type

from IPython.display import display as ipy_display

# Buat logger untuk modul ini
logger = logging.getLogger(__name__)

T = TypeVar('T')

def display_ui_module(
    module: Any,
    module_name: str,
    auto_display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Menampilkan modul UI dengan penanganan error yang konsisten.

    Args:
        module: Instance modul UI yang akan ditampilkan
        module_name: Nama modul untuk keperluan logging
        auto_display: Jika True, akan memanggil method display_ui() pada modul
        **kwargs: Argumen tambahan yang akan diteruskan ke display_ui()

    Returns:
        Dict yang berisi hasil operasi display_ui() jika auto_display=True,
        atau None jika auto_display=False

    Raises:
        RuntimeError: Jika terjadi kesalahan saat menampilkan UI
    """
    try:
        logger.debug(f"Membuat dan menampilkan {module_name} UI")
        
        if auto_display:
            logger.debug(f"Displaying {module_name} UI...")
            if not hasattr(module, 'display_ui'):
                error_msg = f"Modul {module_name} tidak memiliki method display_ui()"
                logger.error(error_msg)
                raise AttributeError(error_msg)
                
            display_result = module.display_ui(**kwargs)
            if not display_result.get('success', False):
                error_msg = display_result.get('message', f'Gagal menampilkan {module_name} UI')
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            logger.debug(f"✅ {module_name} UI displayed successfully")
            return display_result
        else:
            logger.debug(f"✅ {module_name} UI module created (auto-display disabled)")
            # Only display the module object when auto_display is disabled
            ipy_display(module)
            return None
        
    except Exception as e:
        error_msg = f"Gagal membuat dan menampilkan {module_name} UI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

def create_and_display_ui(
    module_class: Type[T],
    module_name: str,
    config: Optional[Dict] = None,
    auto_display: bool = True,
    **kwargs
) -> Optional[T]:
    """
    Factory function untuk membuat dan menampilkan modul UI.

    Args:
        module_class: Kelas modul UI yang akan diinstantiasi
        module_name: Nama modul untuk keperluan logging
        config: Konfigurasi untuk modul
        auto_display: Jika True, akan memanggil method display_ui() pada modul
        **kwargs: Argumen tambahan yang akan diteruskan ke constructor modul

    Returns:
        Instance modul UI yang sudah dibuat

    Raises:
        RuntimeError: Jika terjadi kesalahan saat membuat atau menampilkan UI
    """
    try:
        logger.debug(f"Membuat instance {module_name}")
        module = module_class(config=config, **kwargs)
        
        display_ui_module(
            module=module,
            module_name=module_name,
            auto_display=auto_display,
            **kwargs
        )
        
        return module
        
    except Exception as e:
        error_msg = f"Gagal membuat {module_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
