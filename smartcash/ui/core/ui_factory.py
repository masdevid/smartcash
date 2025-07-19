"""
UI Factory untuk membuat dan menampilkan modul UI dengan pola standar.

File ini menyediakan factory untuk membuat dan menampilkan komponen UI
menggunakan BaseUIModule sebagai dasar, dengan integrasi error handler inti.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/core/ui_factory.py
"""

from typing import Dict, Any, Optional, Type, Callable, TypeVar, Union
import traceback
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.errors.handlers import get_error_handler, CoreErrorHandler
from smartcash.ui.core.errors.exceptions import UIError
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.error.error_component import create_error_component

T = TypeVar('T', bound=BaseUIModule)

class UIFactory:
    """
    Factory untuk membuat dan menampilkan komponen UI berbasis BaseUIModule.
    
    Factory ini menyediakan cara standar untuk membuat dan menampilkan
    komponen UI dengan penanganan error dan logging yang konsisten.
    """
    
    @classmethod
    def create_module(
        cls,
        module_class: Type[T],
        module_name: str,
        parent_module: str = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> T:
        """
        Membuat instance modul UI baru.
        
        Args:
            module_class: Kelas modul UI yang akan dibuat
            module_name: Nama modul
            parent_module: Nama modul induk (opsional)
            config: Konfigurasi awal untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Instance modul UI yang sudah diinisialisasi
        """
        logger = get_module_logger("smartcash.ui.core.ui_factory")
        
        try:
            # Buat instance modul
            module = module_class(
                module_name=module_name,
                parent_module=parent_module,
                **kwargs
            )
            
            # Inisialisasi konfigurasi jika disediakan
            if config and hasattr(module, 'update_config'):
                module.update_config(config)
                
            logger.debug(f"✅ Modul {module.full_module_name} berhasil dibuat")
            return module
            
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                f"Gagal membuat modul {module_name}",
                e,
                logger=logger,
                context={
                    'module_class': module_class.__name__,
                    'module_name': module_name,
                    'parent_module': parent_module
                }
            )
            
            # Create and display error component
            error_ui = create_error_component(
                error_message=f"Gagal membuat modul {module_name}",
                traceback=''.join(traceback.format_exception(type(e), e, e.__traceback__)),
                title="Error Membuat Modul",
                error_type="error"
            )
            display(error_ui['widget'])
            
            raise UIError(f"Gagal membuat modul {module_name}: {str(e)}") from e
    
    @classmethod
    def create_and_display(
        cls,
        module_class: Type[T],
        module_name: str,
        parent_module: str = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Membuat dan menampilkan modul UI dalam satu langkah.
        
        Args:
            module_class: Kelas modul UI yang akan dibuat
            module_name: Nama modul
            parent_module: Nama modul induk (opsional)
            config: Konfigurasi awal untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Informasi modul atau None jika berhasil ditampilkan
        """
        try:
            # Buat modul
            module = cls.create_module(
                module_class=module_class,
                module_name=module_name,
                parent_module=parent_module,
                config=config,
                **kwargs
            )
            
            # Tampilkan UI
            if hasattr(module, 'display_ui'):
                display_result = module.display_ui()
                if display_result and not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    error_handler = get_error_handler()
                    error_handler.handle_error(
                        f"Gagal menampilkan modul {module.module_name if hasattr(module, 'module_name') else 'unknown'}",
                        Exception(error_msg),
                        logger=logger,
                        context={
                            'module_class': module.__class__.__name__,
                            'module_name': getattr(module, 'module_name', 'unknown'),
                            'parent_module': getattr(module, 'parent_module', None)
                        }
                    )
                    
                    # Create and display error component
                    error_ui = create_error_component(
                        error_message=f"Gagal menampilkan modul {getattr(module, 'module_name', 'unknown')}",
                        traceback=''.join(traceback.format_exception(type(Exception(error_msg)), Exception(error_msg), Exception(error_msg).__traceback__)),
                        title="Error Menampilkan Modul",
                        error_type="error"
                    )
                    display(error_ui['widget'])
                    
                    return None
                    
                return module.get_module_info() if hasattr(module, 'get_module_info') else {}
                
            return None
            
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                f"Gagal menampilkan modul {module_name}",
                e,
                logger=logger,
                context={
                    'module_class': module_class.__name__,
                    'module_name': module_name,
                    'parent_module': parent_module
                }
            )
            
            # Create and display error component
            error_ui = create_error_component(
                error_message=f"Gagal menampilkan modul {module_name}",
                traceback=''.join(traceback.format_exception(type(e), e, e.__traceback__)),
                title="Error Menampilkan Modul",
                error_type="error"
            )
            display(error_ui['widget'])
            
            return {'error': str(e), 'success': False}
    
    @classmethod
    def create_display_function(
        cls,
        module_class: Type[T],
        function_name: str = None,
        module_name: str = None
    ) -> Callable:
        """
        Membuat fungsi display untuk modul UI.
        
        Args:
            module_class: Kelas modul UI
            function_name: Nama fungsi yang diinginkan (opsional)
            module_name: Nama modul (jika berbeda dengan nama kelas)
            
        Returns:
            Fungsi yang dapat dipanggil untuk menampilkan modul UI
        """
        mod_name = module_name or module_class.__name__.lower()
        
        def display_function(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
            """Menampilkan modul UI dengan konfigurasi yang diberikan."""
            return cls.create_and_display(
                module_class=module_class,
                module_name=mod_name,
                config=config,
                **kwargs
            )
        
        # Atur metadata fungsi
        display_function.__name__ = function_name or f"show_{mod_name}"
        display_function.__doc__ = f"""
        Menampilkan antarmuka {mod_name}.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Parameter tambahan untuk inisialisasi modul
        """
        
        return display_function
    
    @classmethod
    def _handle_error(
        cls,
        error: Exception,
        module_name: str = "ui_factory",
        operation: str = "unknown",
        ui_components: Optional[Dict[str, Any]] = None,
        **context
    ) -> None:
        """
        Menangani error menggunakan CoreErrorHandler.
        
        Args:
            error: Exception yang terjadi
            module_name: Nama modul tempat error terjadi
            operation: Operasi yang sedang dilakukan saat error
            ui_components: Komponen UI untuk menampilkan error
            **context: Konteks tambahan untuk error
        """
        handler = get_error_handler()
        
        # Tambahkan konteks default
        error_context = {
            'component': module_name,
            'operation': operation,
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            **context
        }
        
        # Handle error dengan UI components jika tersedia
        if ui_components:
            handler._ui_components = ui_components
            
        handler.handle_error(
            error_msg=str(error),
            level='ERROR',
            exc_info=True,
            create_ui_error=bool(ui_components),
            **error_context
        )

# Fungsi utilitas untuk kemudahan penggunaan
def create_ui_display(module_class: Type[T], **kwargs) -> Callable:
    """
    Membuat fungsi display untuk modul UI.
    
    Contoh penggunaan:
        from my_module import MyUIModule
        show_my_ui = create_ui_display(MyUIModule, module_name="my_module")
        show_my_ui(config=my_config)
    
    Args:
        module_class: Kelas modul UI
        **kwargs: Argumen tambahan untuk UIFactory.create_display_function
        
    Returns:
        Fungsi display untuk modul UI
    """
    return UIFactory.create_display_function(module_class, **kwargs)
