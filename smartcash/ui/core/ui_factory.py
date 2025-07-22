"""
UI Factory untuk membuat dan menampilkan modul UI dengan pola standar.

File ini menyediakan factory untuk membuat dan menampilkan komponen UI
menggunakan BaseUIModule sebagai dasar, dengan integrasi error handler inti.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/core/ui_factory.py
"""

import io
import sys
import traceback
from typing import Dict, Any, Optional, Type, Callable, TypeVar
import weakref
import threading
from contextlib import contextmanager

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.errors.handlers import get_error_handler
from smartcash.ui.core.errors.exceptions import UIError
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.error.error_component import create_error_component

T = TypeVar('T', bound=BaseUIModule)

class UIFactory:
    """
    Factory untuk membuat dan menampilkan komponen UI berbasis BaseUIModule.
    
    Factory ini menyediakan cara standar untuk membuat dan menampilkan
    komponen UI dengan penanganan error dan logging yang konsisten.
    
    Implements cache lifecycle management as per optimization.md:
    1. Creation: Components cached on first successful creation
    2. Validation: Cache validated before reuse to ensure integrity
    3. Invalidation: Cache cleared on errors or explicit reset
    4. Cleanup: Factory handles both instance and global cache clearing
    """
    
    # Cache lifecycle management (optimization.md compliance)
    _module_cache: Dict[str, weakref.ref] = {}
    _cache_lock = threading.RLock()
    _cache_validation_enabled = True
    
    @classmethod
    def _get_cache_key(cls, module_class: Type[T], module_name: str, parent_module: str = None) -> str:
        """Generate unique cache key for module instance."""
        base_key = f"{module_class.__name__}:{module_name}"
        return f"{parent_module}:{base_key}" if parent_module else base_key
    
    @classmethod
    def _validate_cached_module(cls, module: BaseUIModule) -> bool:
        """Validate cached module integrity before reuse."""
        if not cls._cache_validation_enabled:
            return True
            
        try:
            # Check if module has required attributes and methods
            required_attrs = ['module_name', 'initialize', 'display_ui']
            for attr in required_attrs:
                if not hasattr(module, attr):
                    return False
            
            # Check if module is properly initialized
            if hasattr(module, '_initialized') and not module._initialized:
                return False
                
            return True
            
        except Exception:
            return False
    
    @classmethod
    def _cache_module(cls, cache_key: str, module: BaseUIModule) -> None:
        """Cache module instance using weak reference."""
        with cls._cache_lock:
            def cleanup_callback(ref):
                # Remove from cache when module is garbage collected
                cls._module_cache.pop(cache_key, None)
            
            cls._module_cache[cache_key] = weakref.ref(module, cleanup_callback)
    
    @classmethod
    def _get_cached_module(cls, cache_key: str) -> Optional[BaseUIModule]:
        """Retrieve and validate cached module instance."""
        with cls._cache_lock:
            if cache_key not in cls._module_cache:
                return None
                
            module_ref = cls._module_cache[cache_key]
            module = module_ref()
            
            # Clean up if module was garbage collected
            if module is None:
                del cls._module_cache[cache_key]
                return None
            
            # Validate cached module integrity
            if not cls._validate_cached_module(module):
                del cls._module_cache[cache_key]
                return None
                
            return module
    
    @classmethod
    def _invalidate_cache(cls, cache_key: str = None) -> None:
        """Invalidate cache (specific key or all)."""
        with cls._cache_lock:
            if cache_key:
                cls._module_cache.pop(cache_key, None)
            else:
                cls._module_cache.clear()
    
    @classmethod
    def clear_all_cache(cls) -> None:
        """Clear all cached module instances."""
        cls._invalidate_cache()
    
    @classmethod
    @contextmanager
    def disable_cache_validation(cls):
        """Context manager to temporarily disable cache validation."""
        original_state = cls._cache_validation_enabled
        cls._cache_validation_enabled = False
        try:
            yield
        finally:
            cls._cache_validation_enabled = original_state
    
    @classmethod
    def _suppress_console_errors(cls):
        """Context manager untuk menekan output error standar Python."""
        class SuppressStderr:
            def __enter__(self):
                self.stderr = sys.stderr
                sys.stderr = io.StringIO()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stderr = self.stderr
                return True
                
        return SuppressStderr()
    
    @classmethod
    def _format_error_message(cls, error: Exception, context: Dict[str, Any] = None) -> str:
        """Format pesan error untuk ditampilkan ke pengguna."""
        context = context or {}
        error_type = error.__class__.__name__
        error_msg = str(error)
        
        # Tambahkan konteks tambahan jika tersedia
        context_info = []
        if 'module_name' in context:
            context_info.append(f"module: {context['module_name']}")
        if 'operation' in context:
            context_info.append(f"operation: {context['operation']}")
            
        context_str = f" ({', '.join(context_info)})" if context_info else ""
        
        return f"{error_type}: {error_msg}{context_str}"
    
    @classmethod
    def _handle_ui_error(
        cls,
        error: Exception,
        error_message: str,
        logger,
        context: Optional[Dict[str, Any]] = None,
        show_traceback: bool = True
    ) -> None:
        """Menangani error dengan menampilkan UI error dan mencatat log."""
        context = context or {}
        
        # Log error
        error_handler = get_error_handler()
        error_handler.handle_error(
            error_message,
            error,
            logger=logger,
            context=context
        )
        
        # Format traceback
        tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__)) \
            if show_traceback else None
            
        # Tampilkan error UI
        error_ui = create_error_component(
            error_message=cls._format_error_message(error, context),
            traceback=tb_str,
            title=error_message,
            error_type="error"
        )
        
        # Tekan output error standar dan tampilkan UI error
        with cls._suppress_console_errors():
            display(error_ui['widget'])
    
    @classmethod
    def create_module(
        cls,
        module_class: Type[T],
        module_name: str,
        parent_module: str = None,
        config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True,
        **kwargs
    ) -> T:
        """
        Membuat instance modul UI baru dengan cache lifecycle management.
        
        Args:
            module_class: Kelas modul UI yang akan dibuat
            module_name: Nama modul
            parent_module: Nama modul induk (opsional)
            config: Konfigurasi awal untuk modul
            enable_cache: Enable caching for this module instance
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Instance modul UI yang sudah diinisialisasi
            
        Raises:
            UIError: Jika terjadi kesalahan saat membuat modul
        """
        logger = get_module_logger("smartcash.ui.core.ui_factory")
        cache_key = cls._get_cache_key(module_class, module_name, parent_module)
        
        try:
            # Try to get cached module first (if caching enabled)
            if enable_cache:
                cached_module = cls._get_cached_module(cache_key)
                if cached_module is not None:
                    # Update config if provided
                    if config and hasattr(cached_module, 'update_config'):
                        cached_module.update_config(config)
                    
                    # Minimal logging for performance (optimization.md)
                    if hasattr(cached_module, 'log_debug'):
                        cached_module.log_debug(f"🔄 Cache hit: Menggunakan instance {getattr(cached_module, 'full_module_name', module_name)}")
                    
                    return cached_module
            
            # Create new instance if not cached or caching disabled
            with cls._suppress_console_errors():
                module = module_class(
                    module_name=module_name,
                    parent_module=parent_module,
                    **kwargs
                )
                
                # Initialize module
                if hasattr(module, 'initialize'):
                    init_result = module.initialize()
                    if init_result is False:
                        raise RuntimeError("Module initialization failed")
                
                # Update config if provided
                if config and hasattr(module, 'update_config'):
                    module.update_config(config)
                
                # Cache successful creation (if enabled)
                if enable_cache:
                    cls._cache_module(cache_key, module)
                
                # Informative success message (optimization.md)
                full_name = getattr(module, 'full_module_name', module_name)
                if hasattr(module, 'log_info'):
                    module.log_info(f"✅ Created {full_name} (cached: {enable_cache})")
                else:
                    logger.info(f"✅ Created {full_name}")
                    
                return module
            
        except Exception as e:
            # Invalidate cache on error (optimization.md)
            if enable_cache:
                cls._invalidate_cache(cache_key)
            
            context = {
                'module_class': module_class.__name__,
                'module_name': module_name,
                'parent_module': parent_module,
                'operation': 'create_module',
                'cache_key': cache_key
            }
            
            cls._handle_ui_error(
                error=e,
                error_message=f"Gagal membuat modul {module_name}",
                logger=logger,
                context=context
            )
            
            raise UIError(f"Gagal membuat modul {module_name}: {str(e)}") from e
    
    @classmethod
    def create_and_display(
        cls,
        module_class: Type[T],
        module_name: str,
        parent_module: str = None,
        config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Membuat dan menampilkan modul UI dalam satu langkah dengan caching support.
        
        Args:
            module_class: Kelas modul UI yang akan dibuat
            module_name: Nama modul
            parent_module: Nama modul induk (opsional)
            config: Konfigurasi awal untuk modul
            enable_cache: Enable caching for this module instance
            **kwargs: Argumen tambahan untuk inisialisasi modul
            
        Returns:
            Informasi modul atau None jika berhasil ditampilkan
            
        Notes:
            - Menekan output error standar Python
            - Menampilkan error menggunakan komponen UI yang lebih informatif
            - Implements cache lifecycle management per optimization.md
        """
        logger = get_module_logger("smartcash.ui.core.ui_factory")
        cache_key = cls._get_cache_key(module_class, module_name, parent_module)
        
        try:
            with cls._suppress_console_errors():
                # Create module with caching support
                module = cls.create_module(
                    module_class=module_class,
                    module_name=module_name,
                    parent_module=parent_module,
                    config=config,
                    enable_cache=enable_cache,
                    **kwargs
                )
                
                # Display UI
                if hasattr(module, 'display_ui'):
                    display_result = module.display_ui()
                    if display_result and not display_result.get('success', False):
                        error_msg = display_result.get('message', 'Gagal menampilkan UI')
                        error = Exception(error_msg)
                        
                        # Invalidate cache on display error
                        if enable_cache:
                            cls._invalidate_cache(cache_key)
                        
                        cls._handle_ui_error(
                            error=error,
                            error_message=f"Gagal menampilkan modul {getattr(module, 'module_name', 'unknown')}",
                            logger=logger,
                            context={
                                'module_class': module.__class__.__name__,
                                'module_name': getattr(module, 'module_name', 'unknown'),
                                'parent_module': getattr(module, 'parent_module', None),
                                'operation': 'display_ui',
                                'cache_key': cache_key
                            }
                        )
                        
                        return None
                        
                    return module.get_module_info() if hasattr(module, 'get_module_info') else {}
                    
                return None
                
        except Exception as e:
            # Invalidate cache on any error
            if enable_cache:
                cls._invalidate_cache(cache_key)
                
            cls._handle_ui_error(
                error=e,
                error_message=f"Gagal menampilkan modul {module_name}",
                logger=logger,
                context={
                    'module_class': module_class.__name__,
                    'module_name': module_name,
                    'parent_module': parent_module,
                    'operation': 'create_and_display',
                    'cache_key': cache_key
                }
            )
            
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
