"""
File: smartcash/ui/setup/env_config/env_config_initializer.py

Environment Configuration Initializer Module - Refactored dengan arsitektur baru.

Modul ini menyediakan class EnvConfigInitializer yang bertanggung jawab untuk
inisialisasi dan manajemen environment configuration UI. Menggunakan arsitektur
baru dari smartcash/ui/core untuk konsistensi dan maintainability.

Fitur Utama:
- Implementasi ModuleInitializer untuk consistent initialization
- Manajemen lifecycle environment configuration UI components
- Koordinasi konfigurasi dan workflow menggunakan handlers terbaru
- Error handling dan user feedback yang lebih baik
- Support untuk programmatic dan interactive usage patterns

Contoh:
    >>> initializer = EnvConfigInitializer()
    >>> ui = initializer.initialize()
    >>> display(ui)
"""

from smartcash.ui.core.shared.logger import get_enhanced_logger
from typing import Dict, Any, Optional
from pathlib import Path

# Import base classes dari arsitektur baru
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.handlers.ui_handler import UIHandler
from smartcash.ui.core.shared.logger import get_module_logger

# Import handlers yang sudah direfactor
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler

# Import UI components
from smartcash.ui.setup.env_config.components.env_config_ui import EnvConfigUI

# Import configs
from smartcash.ui.setup.env_config.configs.defaults import DEFAULT_CONFIG


class EnvConfigInitializer(ModuleInitializer):
    """Environment configuration UI initializer menggunakan arsitektur baru.
    
    Class ini bertanggung jawab untuk inisialisasi dan manajemen environment
    configuration UI. Mengikuti ModuleInitializer pattern untuk konsistensi
    dengan UI initializers lainnya dalam aplikasi.
    
    Proses inisialisasi mengikuti tahapan berikut:
    1. Pre-initialization checks
    2. Create UI components
    3. Setup handlers dengan dependency injection
    4. Post-initialization checks dan syncing
    """
    
    def __init__(self):
        """Initialize environment configuration initializer.
        
        Raises:
            RuntimeError: Jika environment manager initialization gagal
        """
        try:
            # Initialize dengan module info
            super().__init__(
                module_name='env_config',
                parent_module='setup'
            )
            
            # Initialize environment manager
            from smartcash.common.environment import get_environment_manager
            self._env_manager = get_environment_manager()
            
            # Initialize handlers storage
            self._handlers = {}
            
            self.logger.info("🔧 Environment configuration initializer siap")
            
        except Exception as e:
            error_msg = f"❌ Gagal initialize environment manager: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration untuk environment setup.
        
        Returns:
            Default configuration dictionary
        """
        return DEFAULT_CONFIG.copy()
    
    def pre_initialize_checks(self):
        """Perform pre-initialization checks."""
        self.logger.info("🔍 Melakukan pre-initialization checks...")
        
        # Check environment manager
        if not self._env_manager:
            raise RuntimeError("Environment manager tidak tersedia")
            
        # Check paths
        required_paths = [
            Path.home(),
            Path.cwd()
        ]
        
        for path in required_paths:
            if not path.exists():
                self.logger.warning(f"⚠️ Path tidak ditemukan: {path}")
        
        self.logger.info("✅ Pre-initialization checks selesai")
    
    def create_ui_components(self) -> Dict[str, Any]:
        """Create UI components untuk environment configuration.
        
        Returns:
            Dictionary berisi UI components
        """
        self.logger.info("🎨 Membuat UI components...")
        
        try:
            # Create main UI component
            env_config_ui = EnvConfigUI()
            ui_components = env_config_ui.create_ui()
            
            self.logger.info("✅ UI components berhasil dibuat")
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal create UI components: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def setup_handlers(self):
        """Setup handlers dengan dependency injection pattern.
        
        Menggunakan config sebagai class property sesuai arsitektur baru.
        """
        self.logger.info("🔧 Setup handlers...")
        
        try:
            # Initialize config handler dulu
            config_handler = ConfigHandler()
            config_handler.config = self.config
            
            # Initialize setup handler dengan config handler
            setup_handler = SetupHandler()
            setup_handler.config = self.config
            
            # Initialize main env config handler
            env_config_handler = EnvConfigHandler(
                ui_components=self.ui_components
            )
            env_config_handler.config = self.config
            
            # Setup dependency injection
            env_config_handler.setup_dependencies(
                config_handler=config_handler,
                setup_handler=setup_handler
            )
            
            # Store handlers
            self._handlers = {
                'env_config': env_config_handler,
                'config': config_handler,
                'setup': setup_handler
            }
            
            self.logger.info("✅ Handlers berhasil di-setup")
            
        except Exception as e:
            error_msg = f"❌ Gagal setup handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def post_initialization_checks(self):
        """Perform post-initialization checks dan syncing.
        
        Method ini dipanggil setelah semua inisialisasi selesai untuk melakukan
        final checks atau updates ke UI state, termasuk syncing config
        templates jika drive mounted dan setup complete.
        """
        try:
            self.logger.info("🔍 Melakukan post-initialization checks...")
            
            # Get setup handler
            setup_handler = self._handlers.get('setup')
            if not setup_handler:
                self.logger.warning("⚠️ Setup handler tidak tersedia")
                return
            
            # Perform initial status check
            setup_handler.perform_initial_status_check(self.ui_components)
            
            # Check if we should sync config templates
            if setup_handler.should_sync_config_templates():
                self.logger.info("📋 Drive mounted dan setup complete, syncing config templates...")
                
                # Sync dengan UI updates enabled
                setup_handler.sync_config_templates(
                    force_overwrite=False,
                    update_ui=True,
                    ui_components=self.ui_components
                )
            
            self.logger.info("✅ Post-initialization checks selesai")
            
        except Exception as e:
            error_msg = f"❌ Error during post-initialization checks: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Don't raise, just log error
    
    @property
    def handlers(self) -> Dict[str, Any]:
        """Get handlers dictionary.
        
        Returns:
            Dictionary berisi handlers
        """
        return self._handlers
    
    def get_handler(self, handler_name: str) -> Optional[Any]:
        """Get specific handler by name.
        
        Args:
            handler_name: Nama handler yang diinginkan
            
        Returns:
            Handler instance atau None jika tidak ditemukan
        """
        return self._handlers.get(handler_name)


def initialize_env_config_ui(config: Dict[str, Any] = None, **kwargs) -> Any:
    """Initialize dan return environment configuration UI.
    
    Ini adalah main entry point untuk environment configuration UI.
    Membuat instance EnvConfigInitializer dan initialize dengan config yang diberikan.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments untuk initializer
        
    Returns:
        Initialized UI widget dari EnvConfigInitializer
    """
    # Get module logger
    logger = get_enhanced_logger('smartcash.ui.setup.env_config')
    logger.debug("🚀 Initializing environment configuration UI")
    
    try:
        # Create dan initialize initializer
        initializer = EnvConfigInitializer()
        return initializer.initialize(config=config, **kwargs)
        
    except Exception as e:
        logger.error(f"❌ Gagal initialize env config UI: {str(e)}", exc_info=True)
        # Return error UI component
        from smartcash.ui.components.error_display import create_error_display
        return create_error_display(f"Gagal initialize environment configuration: {str(e)}")