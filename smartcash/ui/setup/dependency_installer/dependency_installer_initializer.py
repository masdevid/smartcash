"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Fixed dependency installer menggunakan CommonInitializer pattern dengan pengecekan instalasi setelah UI terender
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.setup.dependency_installer.handlers.setup_handlers import setup_dependency_installer_handlers
from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui
from smartcash.common.config.manager import SimpleConfigManager as ConfigManager, get_config_manager as _get_config_manager
from smartcash.common.environment import get_environment_manager as _get_environment_manager


class DependencyInstallerInitializer(CommonInitializer):
    """Fixed dependency installer menggunakan CommonInitializer pattern dengan pengecekan instalasi setelah UI terender"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi default untuk dependency installer dengan pendekatan one-liner"""
        return {
            'auto_install': False,
            'selected_packages': ['yolov5_req', 'smartcash_req', 'torch_req'],
            'custom_packages': '',
            'validate_after_install': True,
            'delay_analysis': True,  # Flag untuk menunda analisis sampai UI terender
            'suppress_logs': True,  # Tekan log selama inisialisasi
            'hide_progress': True,  # Sembunyikan progress selama inisialisasi
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical component keys yang harus ada"""
        return ['ui', 'install_button', 'status', 'log_output', 'progress_container']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk dependency installer"""
        ui_components = create_dependency_installer_ui(env, config)
        ui_components.update({'module_name': 'DEPS', 'dependency_installer_initialized': False})
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk dependency installer dengan penanganan error yang lebih baik"""
        try:
            # Tambahkan flag untuk mencegah log duplikat
            ui_components['dependency_installer_initialized'] = True
            
            # Setup handlers dengan penanganan error yang lebih baik
            ui_components = setup_dependency_installer_handlers(ui_components, env, config)
            ui_components['handlers_setup'] = True
            
            # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message']("✅ Handlers berhasil disetup", "success")
        except Exception as e:
            # Tangkap error dan log dengan cara yang lebih baik
            error_message = f"❌ Handlers setup failed: {str(e)}"
            
            # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](error_message, "error")
            else:
                # Fallback ke logger jika log_message tidak tersedia
                logger = ui_components.get('logger', self.logger)
                if logger: logger.error(error_message)
            
            # Tandai setup gagal tapi jangan gagalkan inisialisasi
            ui_components['handlers_setup'] = False
        
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validasi tambahan untuk memastikan install button berfungsi dengan pendekatan one-liner"""
        # Validasi komponen UI dengan pendekatan one-liner
        return all([
            'install_button' in ui_components,
            hasattr(ui_components['install_button'], 'on_click'),
            ui_components['install_button']._click_handlers.callbacks,
            'log_output' in ui_components,
            'progress_container' in ui_components,
        ])
        
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Hook yang dijalankan setelah inisialisasi selesai untuk setup UI dan reset handler"""
        from IPython.display import display
        
        # Tampilkan UI
        if 'ui' in ui_components: display(ui_components['ui'])
        
        # Aktifkan log setelah UI terender
        ui_components['suppress_logs'] = False
        
        # Setup reset log dan tampilkan progress handler untuk tombol instalasi
        if 'install_button' in ui_components and hasattr(ui_components['install_button'], 'on_click'):
            original_handler = next(iter(ui_components['install_button']._click_handlers.callbacks), None)
            
            if original_handler:
                # Hapus handler asli
                ui_components['install_button']._click_handlers.callbacks.clear()
                
                # Tambahkan handler baru yang mereset log dan menampilkan progress
                def enhanced_click_handler(b):
                    # Reset log output
                    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
                        ui_components['log_output'].clear_output()
                    
                    # Tampilkan progress container
                    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                        ui_components['progress_container'].layout.visibility = 'visible'
                    
                    # Panggil handler asli
                    original_handler(b)
                
                ui_components['install_button'].on_click(enhanced_click_handler)
        
        # Log status inisialisasi jika tersedia
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("✅ Dependency installer UI berhasil diinisialisasi", "success")
        
        # Jalankan analisis tertunda jika ada
        if config.get('delay_analysis', True) and 'run_delayed_analysis' in ui_components and callable(ui_components['run_delayed_analysis']):
            # Tampilkan progress container untuk analisis
            if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.visibility = 'visible'
            
            # Jalankan analisis tertunda
            ui_components['run_delayed_analysis']()
            
            # Sembunyikan progress container setelah analisis selesai jika tidak ada instalasi otomatis
            if not config.get('auto_install', False) and 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.visibility = 'hidden'
            
        return ui_components
        
    def _get_return_value(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Mendapatkan nilai return yang akan ditampilkan di notebook dengan pendekatan one-liner"""
        from smartcash.ui.utils.silent_wrapper import SilentDictWrapper
        
        # Bungkus ui_components dengan SilentDictWrapper untuk mencegah output otomatis
        return SilentDictWrapper(ui_components)


# Public API
def initialize_dependency_installer(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Inisialisasi dan tampilkan UI dependency installer dengan pendekatan one-liner"""
    from IPython.display import clear_output
    
    # Clear output terlebih dahulu untuk menghindari masalah rendering
    clear_output(wait=True)
    
    # Siapkan config dasar dengan pendekatan one-liner
    config = config or {}
    config.update({'suppress_logs': True, 'hide_progress': True})  # Tambahkan flag untuk menekan log dan progress
    
    # Inisialisasi UI dengan CommonInitializer pattern dan pendekatan one-liner
    return DependencyInstallerInitializer().initialize(config, env)

# Alias fungsi yang sudah ada untuk konsistensi API dengan pendekatan one-liner
get_config_manager, get_environment_manager = _get_config_manager, _get_environment_manager