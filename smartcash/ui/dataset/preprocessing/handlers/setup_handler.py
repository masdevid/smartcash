"""
File: smartcash/ui/dataset/preprocessing/handlers/setup_handler.py
Deskripsi: Handler untuk inisialisasi dan setup komponen preprocessing dengan Environment dan Config Manager baru
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class SetupHandler:
    """Handler untuk setup dan inisialisasi komponen preprocessing."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi setup handler dengan komponen UI."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.setup')
        self.environment_manager = get_environment_manager()
        self.config_manager = get_config_manager()
        
    def initialize_preprocessing_module(self, env=None, config=None) -> Dict[str, Any]:
        """
        Inisialisasi lengkap modul preprocessing.
        
        Args:
            env: Environment manager (opsional, akan menggunakan singleton jika None)
            config: Konfigurasi tambahan (opsional)
            
        Returns:
            Dictionary UI components yang telah diinisialisasi
        """
        try:
            self.logger.info("🚀 Memulai inisialisasi modul preprocessing")
            
            # Phase 1: Environment Setup
            self._setup_environment(env)
            
            # Phase 2: Logger Bridge Setup
            self._setup_logger_bridge()
            
            # Phase 3: Config Loading & Integration
            self._setup_config_integration(config)
            
            # Phase 4: Drive Storage Setup
            self._setup_drive_storage()
            
            # Phase 5: Service Integration
            self._setup_service_integration()
            
            # Phase 6: Handler Registration
            self._register_all_handlers()
            
            # Phase 7: Observer Setup
            self._setup_observer_system()
            
            # Phase 8: Final Validation
            self._validate_setup()
            
            self.logger.success("✅ Modul preprocessing berhasil diinisialisasi")
            return self.ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error saat inisialisasi modul preprocessing: {str(e)}")
            return self._create_error_ui(str(e))
    
    def _setup_environment(self, env=None) -> None:
        """Setup environment manager dan deteksi lingkungan."""
        try:
            # Gunakan environment manager yang sudah ada atau buat baru
            if env:
                self.environment_manager = env
            
            # Log environment info
            env_info = self.environment_manager.get_system_info()
            self.logger.info(f"🌍 Environment: {env_info['environment']}")
            self.logger.info(f"📁 Base Directory: {env_info['base_directory']}")
            
            if env_info.get('drive_mounted', False):
                self.logger.info(f"☁️ Google Drive: {env_info['drive_path']}")
            else:
                self.logger.info("📂 Google Drive: Tidak terpasang")
            
            # Simpan ke UI components
            self.ui_components['environment_manager'] = self.environment_manager
            self.ui_components['environment_info'] = env_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup environment: {str(e)}")
            # Fallback ke environment minimal
            self.ui_components['environment_info'] = {'environment': 'Unknown'}
    
    def _setup_logger_bridge(self) -> None:
        """Setup logger bridge untuk integrasi UI dengan common logger."""
        try:
            # Buat UI logger bridge
            logger_bridge = create_ui_logger_bridge(
                self.ui_components, 
                namespace='smartcash.ui.dataset.preprocessing'
            )
            
            # Simpan ke UI components
            self.ui_components['logger'] = logger_bridge
            self.ui_components['logger_bridge'] = logger_bridge
            self.ui_components['preprocessing_initialized'] = True
            
            self.logger.info("🔗 Logger bridge berhasil disetup")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup logger bridge: {str(e)}")
            # Fallback ke logger biasa
            self.ui_components['logger'] = self.logger
    
    def _setup_config_integration(self, additional_config=None) -> None:
        """Setup integrasi dengan config manager."""
        try:
            # Load preprocessing config dari config manager
            full_config = self.config_manager.get_config()
            preprocessing_config = full_config.get('preprocessing', {})
            
            # Merge dengan additional config jika ada
            if additional_config and isinstance(additional_config, dict):
                preprocessing_config.update(additional_config)
            
            # Simpan ke UI components
            self.ui_components['config'] = preprocessing_config
            self.ui_components['config_manager'] = self.config_manager
            
            self.logger.info("⚙️ Integrasi config manager berhasil")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup config integration: {str(e)}")
            # Fallback ke config minimal
            self.ui_components['config'] = {}
    
    def _setup_drive_storage(self) -> None:
        """Setup Google Drive storage jika tersedia."""
        try:
            from smartcash.ui.dataset.preprocessing.utils.drive_utils import DriveStorageManager
            
            # Buat drive storage manager
            drive_manager = DriveStorageManager(
                self.ui_components, 
                self.environment_manager
            )
            
            # Setup storage
            storage_info = drive_manager.setup_storage()
            
            # Simpan ke UI components
            self.ui_components['drive_manager'] = drive_manager
            self.ui_components['storage_info'] = storage_info
            
            if storage_info.get('drive_available', False):
                self.logger.info("☁️ Google Drive storage berhasil disetup")
            else:
                self.logger.info("📂 Menggunakan local storage")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup drive storage: {str(e)}")
            # Fallback ke local storage
            self.ui_components['storage_info'] = {'drive_available': False}
    
    def _setup_service_integration(self) -> None:
        """Setup integrasi dengan backend services."""
        try:
            from smartcash.ui.dataset.preprocessing.utils.service_integration import ServiceIntegrator
            
            # Buat service integrator
            service_integrator = ServiceIntegrator(self.ui_components)
            
            # Setup service integration
            service_integrator.setup_integration()
            
            # Simpan ke UI components
            self.ui_components['service_integrator'] = service_integrator
            
            self.logger.info("🔌 Integrasi backend services berhasil")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup service integration: {str(e)}")
    
    def _register_all_handlers(self) -> None:
        """Register semua handlers untuk UI components."""
        try:
            # Register main preprocessing handler
            self._register_main_handler()
            
            # Register config handlers (save/reset)
            self._register_config_handlers()
            
            # Register cleanup handler
            self._register_cleanup_handler()
            
            # Register stop handler
            self._register_stop_handler()
            
            self.logger.info("🔧 Semua handlers berhasil diregistrasi")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error registrasi handlers: {str(e)}")
    
    def _register_main_handler(self) -> None:
        """Register main preprocessing handler."""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.main_handler import setup_main_button
            setup_main_button(self.ui_components)
        except Exception as e:
            self.logger.warning(f"⚠️ Error register main handler: {str(e)}")
    
    def _register_config_handlers(self) -> None:
        """Register config handlers (save/reset)."""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_buttons
            setup_config_buttons(self.ui_components)
        except Exception as e:
            self.logger.warning(f"⚠️ Error register config handlers: {str(e)}")
    
    def _register_cleanup_handler(self) -> None:
        """Register cleanup handler."""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import CleanupHandler
            CleanupHandler(self.ui_components)
        except Exception as e:
            self.logger.warning(f"⚠️ Error register cleanup handler: {str(e)}")
    
    def _register_stop_handler(self) -> None:
        """Register stop handler."""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.stop_handler import setup_stop_button
            setup_stop_button(self.ui_components)
        except Exception as e:
            self.logger.warning(f"⚠️ Error register stop handler: {str(e)}")
    
    def _setup_observer_system(self) -> None:
        """Setup sistem observer untuk komunikasi antar komponen."""
        try:
            # Coba import observer system
            try:
                from smartcash.components.observer.manager_observer import get_observer_manager
                observer_manager = get_observer_manager()
            except ImportError:
                # Fallback ke mock observer
                from smartcash.ui.dataset.preprocessing.utils.ui_observers import MockObserverManager
                observer_manager = MockObserverManager()
                self.logger.info("🔄 Menggunakan mock observer (observer system tidak tersedia)")
            
            # Setup observer manager
            self.ui_components['observer_manager'] = observer_manager
            self.ui_components['observer_group'] = 'preprocessing_ui'
            
            # Register basic observers untuk UI feedback
            self._register_ui_observers(observer_manager)
            
            self.logger.info("👁️ Sistem observer berhasil disetup")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup observer system: {str(e)}")
            # Fallback ke sistem sederhana
            self.ui_components['observer_manager'] = None
    
    def _register_ui_observers(self, observer_manager) -> None:
        """Register observers untuk UI feedback."""
        try:
            # Register progress observer jika tersedia
            if hasattr(observer_manager, 'create_progress_observer'):
                progress_observer = observer_manager.create_progress_observer(
                    event_types=['preprocessing.progress', 'preprocessing.status'],
                    total=100,
                    desc="Preprocessing Progress",
                    group='preprocessing_ui'
                )
                self.ui_components['progress_observer'] = progress_observer
            
            # Register logging observer jika tersedia
            if hasattr(observer_manager, 'create_logging_observer'):
                log_observer = observer_manager.create_logging_observer(
                    event_types=['preprocessing.log', 'preprocessing.error'],
                    log_level='info',
                    group='preprocessing_ui'
                )
                self.ui_components['log_observer'] = log_observer
                
        except Exception as e:
            self.logger.debug(f"ℹ️ Observer registration optional features gagal: {str(e)}")
    
    def _validate_setup(self) -> None:
        """Validasi bahwa setup berhasil dan semua komponen siap."""
        required_components = [
            'logger', 'config_manager', 'environment_manager', 
            'preprocessing_initialized'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in self.ui_components:
                missing_components.append(component)
        
        if missing_components:
            self.logger.warning(f"⚠️ Komponen tidak lengkap: {missing_components}")
        
        # Validasi UI components
        ui_required = ['preprocess_button', 'status_panel']
        missing_ui = [comp for comp in ui_required if comp not in self.ui_components]
        
        if missing_ui:
            self.logger.warning(f"⚠️ UI components tidak lengkap: {missing_ui}")
        
        # Set flags
        self.ui_components['setup_completed'] = len(missing_components) == 0
        self.ui_components['preprocessing_running'] = False
        self.ui_components['cleanup_running'] = False
        self.ui_components['stop_requested'] = False
    
    def _create_error_ui(self, error_message: str) -> Dict[str, Any]:
        """Buat UI error sederhana jika setup gagal."""
        try:
            import ipywidgets as widgets
            from smartcash.ui.utils.constants import COLORS, ICONS
            
            error_ui = widgets.VBox([
                widgets.HTML(f"""
                <div style="padding: 20px; background-color: #f8d7da; 
                           border: 1px solid #f5c6cb; border-radius: 5px;">
                    <h3 style="color: #721c24; margin-top: 0;">
                        {ICONS.get('error', '❌')} Error Inisialisasi Preprocessing
                    </h3>
                    <p style="color: #721c24; margin-bottom: 15px;">
                        Terjadi kesalahan saat menginisialisasi modul preprocessing:
                    </p>
                    <div style="background-color: #fff; padding: 10px; 
                              border-radius: 3px; font-family: monospace;">
                        {error_message}
                    </div>
                    <p style="color: #721c24; margin-top: 15px; margin-bottom: 0;">
                        Silakan restart cell atau hubungi administrator.
                    </p>
                </div>
                """)
            ])
            
            return {
                'ui': error_ui,
                'error': True,
                'error_message': error_message,
                'preprocessing_initialized': False
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat error UI: {str(e)}")
            return {
                'error': True,
                'error_message': error_message,
                'preprocessing_initialized': False
            }
    
    def cleanup_resources(self) -> None:
        """Cleanup resources saat setup handler tidak digunakan lagi."""
        try:
            # Cleanup observer manager
            observer_manager = self.ui_components.get('observer_manager')
            if observer_manager and hasattr(observer_manager, 'unregister_group'):
                observer_manager.unregister_group('preprocessing_ui')
            
            # Cleanup handlers
            for handler_key in ['config_handler', 'cleanup_handler', 'main_handler']:
                if handler_key in self.ui_components:
                    handler = self.ui_components[handler_key]
                    if hasattr(handler, 'cleanup_resources'):
                        handler.cleanup_resources()
            
            self.logger.debug("🧹 Setup handler resources berhasil dibersihkan")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat cleanup setup handler: {str(e)}")

# Factory function untuk membuat setup handler
def create_setup_handler(ui_components: Dict[str, Any]) -> SetupHandler:
    """
    Factory function untuk membuat setup handler.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance SetupHandler yang siap digunakan
    """
    return SetupHandler(ui_components)

# Main function untuk inisialisasi preprocessing
def initialize_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Function utama untuk inisialisasi handlers preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi tambahan (opsional)
        
    Returns:
        Dictionary UI components yang telah diinisialisasi
    """
    # Buat setup handler
    setup_handler = create_setup_handler(ui_components)
    
    # Simpan reference untuk cleanup nanti
    ui_components['setup_handler'] = setup_handler
    
    # Jalankan inisialisasi
    return setup_handler.initialize_preprocessing_module(env, config)